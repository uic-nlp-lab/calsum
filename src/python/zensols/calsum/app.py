"""Summarize text using Component ALignment Abstract Meaning Representation
(CALAMR) alignment.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Set, Any
from dataclasses import dataclass, field
import logging
from itertools import chain
from pathlib import Path
from zensols.util.time import time
from zensols.util import Failure
from zensols.persist import Stash
from zensols.calamr import Resource
from .model import DischargeSummaryGenerator

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Summarize text using Component ALignment Abstract Meaning Representation
    (CALAMR) alignment.

    """
    resource: Resource = field()
    """A stash of :class:`~zensols.calamr.flow.FlowGraphResult."""

    results_dir: Path = field()
    """The directory where the output results are written, then read back for
    analysis reporting.

    """

    _sent_match_factory_stash: Stash = field()
    """The stash that CRUDs instances of :class:`.SentenceMatchSet`."""

    _sent_match_stash: Stash = field()
    """CRUDs instances of :class:`.SentenceMatchSet`."""

    _generator: DischargeSummaryGenerator = field()
    """Generates discharge summaries using the model."""

    def write(self, hadm_id: str):
        """Print an admission.

        :param hadm_id: the hopsital admission ID

        """
        from zensols.clinicamr import AdmissionAmrFeatureDocument
        doc: AdmissionAmrFeatureDocument = \
            self.resource.get_corpus_document(hadm_id)
        doc.write()

    def align(self):
        """Parse and align the admissions."""
        with time('parsed and aligned {cnt} admissions'):
            keys = tuple(self.resource.flow_results_stash.keys())
            cnt = len(keys)

    def align_render(self, hadm_id: str):
        """Render a graph and write it to disk.

        :param hadm_id: the hopsital admission ID

        """
        from zensols.calamr import FlowGraphResult, Resource
        resource: Resource = self.resource
        with time():
            res: FlowGraphResult = resource.align_corpus_document(hadm_id)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            res.render(directory=self.results_dir, display=False)
        #res.write()

    def delete_alignment_errors(self):
        """Remove the failed alignments (missing discharge notes)."""
        self._sent_match_factory_stash.delete_alignment_errors()

    def print_sections(self):
        """Print all sections found in the alignment data helpful to the
        UIHealth dataset.

        """
        # this output is added to ../clinicamr/resources/section-selection.csv
        # for the UI health dataset
        from zensols.mimic import Section
        from zensols.clinicamr import NoteDocument, SectionDocument
        resource: Resource = self.resource
        stash: Stash = resource.flow_results_stash
        for res in stash.values():
            note_docs = chain.from_iterable((
                (res.doc_graph.doc.create_discharge_summary(),),
                res.doc_graph.doc.create_note_antecedents()))
            sec_names: Set[str] = set()
            note_doc: NoteDocument
            for note_doc in note_docs:
                sec_doc: SectionDocument
                for sec_doc in note_doc.create_sections():
                    sec_name: str = Section.header_to_name(sec_doc.name)
                    sec_names.add(sec_name)
        print('\n'.join(sorted(sec_names)))

    def match(self):
        """Match summary and source sentences."""
        stash = self._sent_match_stash
        with time('matched {cnt} graphs'):
            cnt = len(stash)

    def delete_match_errors(self):
        """Remove the failed source/summary matches."""
        stash: Stash = self._sent_match_stash
        logger.info(f'deleting from up to {len(stash)} matches')
        dels: List[str] = []
        for k, v in stash:
            if isinstance(v, Failure):
                logger.info(f'marking failure: {v}')
                dels.append(k)
        for d in dels:
            logger.info(f'deleting: {d}')
            stash.delete(d)

    def generate_summaries(self, hadm_id: str = None):
        """Generate the discarge summaries.

        :param hadm_id: the hospital admission ID

        """
        logger.info('generating summaries..')
        hadm_ids: Tuple[str, ...] = None
        if hadm_id is not None:
            hadm_ids = (hadm_id,)
        with time('generated {cnt} summaries'):
            cnt = len(self._generator.generate(hadm_ids))

    def create_annotation_file(self):
        """Create the informal evaluation spreadsheet."""
        from pathlib import Path
        import pandas as pd
        from zensols.datdesc import DataFrameDescriber
        lk: str = '? 1 out of 5 where 5 is the best/most'
        meta = (('hadm_id', 'the admission ID'),
                ('annotator', 'the human annotator'),
                ('preference', 'Whether you prefer the generated summary' + lk),
                ('readablility', 'Of the data in the generated summary, how readable is it' + lk),
                ('correctness', 'Of the data that is in the generated summary, how correct is it' + lk),
                ('complete', 'How complete is the generated summary'),
                ('sections', 'Of the data in the summary, how well is it sectioned' + lk))
        cols: Tuple[str, ...] = map(lambda t: t[0], meta)
        gen_dir: Path = self._generator.output_dir
        dirs: Tuple[Path, ...] = tuple(gen_dir.iterdir())
        rows: List[Tuple[Any, ...]] = []
        adm_dir: Path
        for i, adm_dir in enumerate(dirs):
            ann: str = 'ac' if i < (len(dirs) / 2) else 'sr'
            rows.append((adm_dir.name, ann, *([None] * (len(meta) - 2))))
        dfd = DataFrameDescriber(
            name='informal-evaluation',
            desc='Informal evaluation of the generated discharge summaries',
            df=pd.DataFrame(rows, columns=cols),
            meta=meta)
        dfd.save_excel(Path('~/Desktop'))
