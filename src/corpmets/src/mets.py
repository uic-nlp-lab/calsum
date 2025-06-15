"""Metrics evaluation.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict, Sequence, Iterable, Any, Optional, Union
from dataclasses import dataclass, field
import sys
import logging
import itertools as it
from io import TextIOBase
from pathlib import Path
import pandas as pd
import evaluate
from zensols.config import Dictable
from zensols.nlp import FeatureDocument
from zensols.mimic import Note, HospitalAdmission, Corpus
from zensols.mimic.regexnote import DischargeSummaryNote

logger = logging.getLogger(__name__)


@dataclass
class AdmMetric(Dictable):
    """A class that produces metrics.

    """
    hadm_id: int = field()
    """The hospital admission ID."""

    ds: Note = field()
    """The discharge summary note."""

    ants: Tuple[Note, ...] = field()
    """The EHR note antecedents (notes written prior to :obj:`ds`)."""

    def get_metrics(self, metrics: Dict[str, Dict[str, Any]]) -> \
            Dict[str, Union[int, float]]:
        """Create the metrics for the discharge summary and EHR note antecedents
        (see :obj:`ants`).  This is done by:

          1. Concatenate all discharge summary sentences (:obj:`ds`).

          2. Concatenate all of the EHR note antecedent sentences (:obj:`ants`)
          sorting by chart date.

          3. Evaluate using the given metric (keys in ``metrics``) using the
          discharge summary sentences as the prediction and the the EHR note
          antecedent sentences as the referecnes.

        """
        def map_metric(k: str, metrics_res: Dict[str, Any]) -> Tuple[str, Any]:
            val = metrics_res[k]
            if isinstance(val, list):
                val = val[0]
            return (k, val)

        # get discharge summary sentences
        ds_doc: FeatureDocument = self.ds.doc
        ds_sents: Tuple[str, ...] = list(map(lambda s: s.norm, ds_doc.sents))
        # get note antecdent sentences of all notes in order of chart date
        ant_sents: List[str] = []
        for ant in sorted(self.ants, key=lambda n: n.chartdate):
            ant_sents.extend(map(lambda s: s.norm, ant.doc.sents))
        row: Dict[str, Any] = {
            'hadm_id': self.hadm_id,
            'ds_sents': len(ds_sents),
            'ds_tokens': ds_doc.token_len,
            'ant_sents': len(ant_sents),
            'ant_tokens': sum(map(lambda d: d.doc.token_len, self.ants))}
        # evaluate using each request metric in ``metrics``
        metric_name: str
        config: Dict[str, Any]
        for metric_name, config in metrics.items():
            keys: Sequence[str] = config['keys']
            params: Optional[Dict[str, Any]] = config.get('params', {})
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'metrics: {metric_name}, params: {params}')
            metric_ev = evaluate.load(metric_name)
            # invoke the evaluation metric computation
            metrics_res: Dict[str, Any] = metric_ev.compute(
                predictions=[' '.join(ds_sents)],
                references=[' '.join(ant_sents)],
                **params)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'evaulation {metric_name}: {metrics_res}')
            row.update(dict(map(lambda k: map_metric(k, metrics_res), keys)))
        return row

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'hadm: {self.hadm_id}', depth, writer)
        self._write_line(f'ds: id={self.ds.row_id}, len={len(self.ds.text)}',
                         depth, writer)
        self._write_line('ants:', depth, writer)
        ant: Note
        for ant in self.ants:
            self._write_line(f'{ant.row_id}: len={len(ant.text)}',
                             depth + 1, writer)

    def __str__(self) -> str:
        return f'{self.hadm_id} ({len(self.ants)} antecedents)'


@dataclass
class MetricsEvaluator(Dictable):
    """Computes metrics between MIMIC-III EHR notes and their respective
    discharge summary.

    """
    corpus: Corpus = field()
    """The MIMIC-III corpus data access object."""

    gen_dir: Path = field()
    """The directory with the generated discharge summaries (only used to get
    hadm IDs).

    """
    metrics: Dict[str, Dict[str, Any]] = field()
    """The metrics to collect.  This is used by :meth:`evaluate` to create the
    returned Pandas dataframe.  The keys are the metrics to use for evaluation,
    and the values are dictionaries that have the following entries:

      * ``keys`` metrics results to add as columns to
      * ``params`` the keword parameters to give to :func:`evaluate.compute`

    """
    def _get_adm_metrics(self) -> Iterable[AdmMetric]:
        hadm_ids: Tuple[int, ...] = tuple(map(
            lambda p: int(p.name), self.gen_dir.iterdir()))
        logger.info(f'found {len(hadm_ids)} admissions')
        hadm_id: int
        for hadm_id in hadm_ids:
            adm: HospitalAdmission = self.corpus.get_hospital_adm_by_id(hadm_id)
            ds: Note = None
            ants: List[Note] = []
            for note in adm:
                if note.category == DischargeSummaryNote.CATEGORY:
                    ds = note
                else:
                    ants.append(note)
            yield AdmMetric(hadm_id, ds, tuple(ants))

    def evaluate(self, limit: int = sys.maxsize) -> pd.DataFrame:
        """Evaluate admission using the method given in
        :meth:`.AdmMetric.get_metrics.`

        :see: :obj:`metrics`

        """
        rows: List[Dict[str, Union[int, float]]] = []
        am: AdmMetric
        for am in it.islice(self._get_adm_metrics(), limit):
            logger.info(f'processing {am}...')
            row: Dict[str, Union[int, float]] = am.get_metrics(self.metrics)
            rows.append(row)
        return pd.DataFrame(rows)
