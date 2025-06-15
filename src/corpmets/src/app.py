"""Application CLI entry point.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
import sys
from pathlib import Path
from mets import MetricsEvaluator

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Evulation metrics harness.

    """
    evaluator: MetricsEvaluator = field()
    """Computes metrics between MIMIC-III EHR notes and their respective
    discharge summary.

    """
    def calculate(self, out_file: Path, limit: int = None):
        """Calculate metrics and save results as a CSV file.

        :param out_file: where to output the metrics

        :param limit: the max number of admissions to process

        """
        import pandas as pd
        if limit is None:
            limit = sys.maxsize
        df: pd.DataFrame = self.evaluator.evaluate(limit=limit)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file)
        logger.info(f'wrote: {out_file}')
