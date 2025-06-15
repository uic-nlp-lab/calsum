"""Report alignment, extractive sentence matches and performance.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict, Set, Any
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
from pathlib import Path
from itertools import chain
import textwrap as tw
import pandas as pd
from sklearn import metrics
from zensols.util.time import time
from zensols.persist import persisted, PersistedWork, Stash, dealloc
from zensols.datdesc import DataFrameDescriber, DataDescriber
from zensols.deeplearn.model import ModelSettings, ModelFacade
from zensols.deeplearn.result import ModelResultManager, ModelResultReporter
from zensols.mimic import Section, Corpus, NoteEventPersister
from zensols.mimicsid import AnnotationResource
from zensols.alsum.graph import ReducedGraph
from zensols.rend import ApplicationFactory as RendApplicationFactory
from zensols.rend import BrowserManager
from .matcher import SentenceMatchSet
from .fig import Figure
from .plots import HeatMapPlot

logger = logging.getLogger(__name__)


@dataclass
class Reporter(object, metaclass=ABCMeta):
    """A base class for reporting data that end up in papers.

    """
    temporary_dir: Path = field()
    """A directory that has temporary files for analysis."""

    def __post_init__(self):
        self._persisted_works: PersistedWork = []

    @property
    @persisted('_browser')
    def browser(self) -> BrowserManager:
        """Get a browser manager for :meth:`render`."""
        return RendApplicationFactory.get_browser_manager()

    def render(self):
        """Get a browser manager to render table data visually."""
        self.browser(self.data_describer)

    @property
    @persisted('_data_describer')
    def data_describer(self) -> DataDescriber:
        """Get the data reported by this reporter instance."""
        return self._get_data_describer()

    @abstractmethod
    def _get_data_describer(self) -> DataDescriber:
        """The subclass gathers and creates the data for this instance."""
        pass

    def save_figures(self):
        """Save any figures / plots the reporter might have."""
        pass

    def _create_contingency(self, df: pd.DataFrame, left_col: str, top_col: str,
                            left_desc: str, top_desc: str, name: str,
                            desc: str) -> DataFrameDescriber:
        """Create a contingency matrix as a dataframe describer.

        :param df: the data that has the two columns to use for the matrix

        :param left_col: the column in ``df`` for the left rows

        :param top_col: the column in ``df`` for the columns

        :param left_desc: the description in ``df`` for the left rows

        :param top_desc: the description in ``df`` for the columns

        :param name: the name to use in the dataframe describer

        :param desc: the description to use in the dataframe describer

        """
        dfc: pd.DataFrame = pd.crosstab(df[left_col], df[top_col])
        dfc.columns = dfc.columns.get_level_values(0)
        dfc.columns.name = None
        dfc.index = dfc.index.to_flat_index()
        dfc.index.name = f'{left_desc} \\ {top_desc}'
        idesc: str = 'Labels are columns, predictions are rows.'
        imeta = {i: i for i in dfc.index}
        imeta[dfc.index.name] = idesc
        return DataFrameDescriber(
            name=name,
            desc=desc,
            df=dfc,
            index_meta=imeta)

    def clear(self):
        """Clear any cached data this reporter used."""
        for work in self._persisted_works:
            work.clear()


@dataclass
class PerformanceReporter(Reporter):
    """A reporter for the performance of the model.

    """
    facade: ModelFacade = field()
    """The facade used for reporting."""

    res_id: str = field(default=None)
    """The result ID used for prediction reporting or ``None`` for the latest.

    """
    model_name: str = field(default=None)
    """The model name to use for reporting.  If not set, it is taken from the
    model settings.

    """
    confusion_figure: Figure = field(default=None)
    """The figure used for the label classification confusion matrix.  The
    figure will be created if set.

    """
    def __post_init__(self):
        super().__post_init__()
        self._predictions_dataframe = PersistedWork(
            self.temporary_dir / 'pred-df.dat', self, mkdir=True)
        self._persisted_works.append(self._predictions_dataframe)

    @property
    @persisted('_model_name_pw')
    def _model_name(self) -> str:
        if hasattr(self, '_model_name_var') and \
           self._model_name_var is not None:
            return self._model_name_var
        else:
            model_settings: ModelSettings = self.facade.executor.model_settings
            return model_settings.normal_model_name

    @_model_name.setter
    def _model_name(self, model_name: str):
        self._model_name_var = model_name

    @property
    @persisted('_predictions_dataframe')
    def predictions(self) -> pd.DataFrame:
        """The predictions of the model."""
        return self.facade.get_predictions(name=self.res_id)

    @property
    def confusion_matrix(self) -> DataFrameDescriber:
        """The confusion matrix of the dataset."""
        df: pd.DataFrame = self.predictions
        labs = df['label'].drop_duplicates().sort_values().to_list()
        cm = metrics.confusion_matrix(df['pred'], df['label'], labels=labs)
        dfc = pd.DataFrame(cm, index=labs, columns=labs)
        dfc.index.name = 'prediction \\ label'
        idesc: str = 'Labels are columns, predictions are rows.'
        imeta = {i: i for i in labs}
        imeta[dfc.index.name] = idesc
        return DataFrameDescriber(
            name=f'{self.model_name}-confusion',
            desc=f'Confusion matrix of dataset: {self.model_name}',
            df=dfc,
            index_meta=imeta)

    def _render_confusion_matrix_figure(self) -> HeatMapPlot:
        """Create a note to section alignment confusion matrix."""
        if self.confusion_figure is not None:
            dfd: DataFrameDescriber = self.confusion_matrix
            return self.confusion_figure.create(
                name=HeatMapPlot,
                format='d',
                title=dfd.desc,
                #x_label_rotation=50,
                dataframe=dfd.df,
                params={'cbar': False})

    @property
    def summary(self) -> DataFrameDescriber:
        rm: ModelResultManager = self.facade.result_manager
        reporter = ModelResultReporter(rm)
        df: pd.DataFrame = reporter.dataframe
        descs: Dict[str, str] = ModelResultReporter.METRIC_DESCRIPTIONS
        meta: List[Tuple[Any, ...]] = []
        for c in df.columns:
            desc: str = descs.get(c)
            if desc is not None:
                meta.append((c, desc))
        return DataFrameDescriber(
            name=f'{self.model_name}-summary',
            desc=f'Results summary for dataset: {self.model_name}',
            df=df,
            meta=meta)

    def _get_data_describer(self) -> DataDescriber:
        return DataDescriber((self.summary,), name=self.model_name)

    def save_figures(self):
        plot: HeatMapPlot = self._render_confusion_matrix_figure()
        if plot is not None:
            self.confusion_figure.save()


PerformanceReporter.model_name = PerformanceReporter._model_name


@dataclass
class CalsumPerformanceReporter(PerformanceReporter):
    human_eval_path: Path = field(default=None)

    @property
    def informal_eval(self) -> DataFrameDescriber:
        df: pd.DataFrame = pd.read_excel(self.human_eval_path)
        df = df.drop(columns='hadm_id annotator'.split())
        mean: pd.DataFrame = df.mean().to_frame()
        mean.columns = ['value']
        mean.insert(0, 'description', mean.index)
        mean = mean.reset_index(drop=True)
        mean = pd.concat((mean, pd.DataFrame(
            (('count', len(df)),),
            columns=mean.columns)))
        return DataFrameDescriber(
            name='informal-eval-gen-summaries',
            desc='Informal evaluation of generated summaries',
            df=mean)

    def _get_data_describer(self) -> DataDescriber:
        dd = super()._get_data_describer()
        dd.describers = (self.informal_eval, *dd.describers)
        return dd


@dataclass
class CalsumDatasetReporter(Reporter):
    """Compute the dataset's statistics for reporting.

    """
    corpus: Corpus = field()
    """The MIMIC-III corpus."""

    facade: ModelFacade = field()
    """The facade used for reporting."""

    graph_stash: Stash = field()
    """The stash that CRUDs :class:`~zensols.alsum.graph.ReducedGraph`."""

    contingency_table: Figure = field()
    """The figure object used to create the confusion matrix."""

    keep_notes: List[str] = field()
    """The note (by category) to keep in each clinical note.  The rest are
    filtered.

    """
    human_eval_path: Path = field()
    """The excel file of the physician informal evaluation."""

    msid_anon_resource: AnnotationResource = field()
    """MedSecId annotation resource."""

    dsprov_stash: Stash = field()
    """The DSProv annotation stash."""

    def __post_init__(self):
        super().__post_init__()
        self._feature_dataframe = PersistedWork(
            self.temporary_dir / 'feat-df.dat', self, mkdir=True)
        self._graph_stats = PersistedWork(
            self.temporary_dir / 'graph-stats-df.dat', self, mkdir=True)
        self._note_counts = PersistedWork(
            self.temporary_dir / 'note-counts-df.dat', self, mkdir=True)
        self._match_admission_note_counts = PersistedWork(
            self.temporary_dir / 'match-adm-note-counts-df.dat',
            self, mkdir=True)
        self._persisted_works.append(self._feature_dataframe)
        self._persisted_works.append(self._graph_stats)
        self._persisted_works.append(self._note_counts)
        self._persisted_works.append(self._match_admission_note_counts)

    @property
    @persisted('_feature_dataframe')
    def feature_dataframe(self):
        """A dataframe of the dataset by split of matched sentences.  This
        matched sentence dataset is created using the Calamr alignments and then
        using the sentence matching algorithm.

        """
        stash: Stash = self.facade.feature_stash
        dfs: List[pd.DataFrame] = []
        with time('created feature dataframe'):
            split: str
            sstash: Stash
            for split, sstash in stash.splits.items():
                sm: SentenceMatchSet
                for sm in sstash.values():
                    df: pd.DataFrame = sm.dataframe
                    df.insert(0, 'split', split)
                    df.insert(0, 'hadm_id', sm.hadm_id)
                    dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def get_admission_counts(self) -> int:
        """Get the total number of matched admissions."""
        df: pd.DataFrame = self.feature_dataframe
        n_adms: int = len(df['hadm_id'].drop_duplicates())
        # 1,330
        return n_adms

    @property
    @persisted('_note_counts')
    def note_counts(self):
        """Counts of notes per admission and across note category used in the
        matched sentence dataset.

        """
        np: NoteEventPersister = self.corpus.note_event_persister
        df: pd.DataFrame = self.feature_dataframe
        rows: List[Tuple[Any, ...]] = []
        cat_cols: List[str, ...] = sorted(self.keep_notes)
        cols: List[str] = 'hadm_id total aligned'.split() + cat_cols
        with time('created note stats dataframe'):
            hadm_id: str
            for hadm_id in df['hadm_id'].drop_duplicates():
                n_notes: int = len(np.get_row_ids_by_hadm_id(int(hadm_id)))
                by_cat: Dict[str, List[int]] = np.get_row_ids_by_category(
                    int(hadm_id), self.keep_notes)
                aligned_notes: int = sum(
                    1 for _ in chain.from_iterable(by_cat.values()))
                row: List[Any] = [hadm_id, n_notes, aligned_notes]
                for col in cat_cols:
                    row.append(len(by_cat.get(col, ())))
                rows.append(row)
            df = pd.DataFrame(rows, columns=cols)
            df = df.rename(columns=dict(
                zip(cat_cols, map(Section.header_to_name, cat_cols))))
        return DataFrameDescriber(
            name='note-counts',
            desc='Counts of notes per admission and across note category',
            df=df,
            meta=(('hadm_id', 'admission ID'),
                  ('total', 'total number of notes for the admission'),
                  ('aligned', 'number of notes used for alignment'),
                  *(map(lambda c: (Section.header_to_name(c),
                                   f'number of {c} notes used for alignment'),
                        cat_cols))))

    @property
    def note_stats(self) -> DataFrameDescriber:
        """Summarized form of :obj:`note_counts`."""
        dfd: DataFrameDescriber = self.note_counts
        df = dfd.df.drop(columns='hadm_id'.split()).sum().T.to_frame()
        df.columns = 'count'.split()
        df.insert(0, 'type', df.index)
        imeta = dict(zip(dfd.meta.index, dfd.meta.iloc[:, 0]))
        df.index.name = 'description'
        imeta[df.index.name] = 'description'
        return DataFrameDescriber(
            name='note-counts',
            desc=dfd.desc,
            df=df,
            meta=(('type', 'type of data or note name'),
                  ('count', 'number of occurances')),
            index_meta=imeta)

    @persisted('_graph_stats')
    def _get_graph_stats(self) -> pd.DataFrame:
        """Return a dataframe with the alignment statistics of all graphs.
        Takes 44 min.

        """
        dfs: List[pd.DataFrame] = []
        hadm_id: str
        rg: ReducedGraph
        with time('created graph stats dataframe'):
            for hadm_id, rg in self.graph_stash.items():
                with dealloc(rg):
                    df: pd.DataFrame = rg.graph_result.stats_df
                    df.insert(0, 'hadm_id', hadm_id)
                    dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    @property
    def graph_stats(self) -> DataFrameDescriber:
        """Reports counts and averages of alignments and nodes."""
        df: pd.DataFrame = self._get_graph_stats()
        rows: List[Tuple[Any, ...]] = []
        comp: str
        for comp in 'source summary'.split():
            dfc = df[df['component'] == comp]
            av: pd.Series = dfc['aligned alignable reentrancies'.split()].mean()
            for col, val in av.items():
                rows.append((f'{comp}_{col}', f'average {col} in the {comp}', val))
        rows.append(('total', 'total number of graphs aligned', len(df)))
        return DataFrameDescriber(
            name='graph-statistics',
            desc='Graph alignment statistics',
            df=pd.DataFrame(rows, columns='name description value'.split()))

    def get_label_stats(self) -> DataFrameDescriber:
        """A dataframe of label counts across splits."""
        df: pd.DataFrame = self.feature_dataframe
        dfs: List[pd.DataFrame] = []
        for split, dfg in df.groupby('split'):
            dfg['smy_sec'] = dfg['smy_sec'].fillna('none')
            dfh = dfg.groupby('smy_sec').agg({'smy_sec': 'count'}).\
                rename(columns={'smy_sec': split}).reset_index().\
                rename(columns={'smy_sec': 'section'}).\
                sort_values('section', ascending=False).reset_index(drop=True)
            dfs.append(dfh)
        df = dfs[1].merge(dfs[0].merge(dfs[2], on='section'), on='section')
        df = df.sort_values('train', ascending=False)
        if 0:
            # proportions of labels across splits
            df['tp'] = df['test'] / df['train']
            df['vp'] = df['validation'] / df['train']
            df['dp'] = df['tp'] - df['vp']
            df = df.sort_values('dp')
        trow: pd.DataFrame = df['train test validation'.split()].sum()
        trow['section'] = 'total'
        df = pd.concat((df, trow.to_frame().T), ignore_index=True)
        return DataFrameDescriber(
            name='section-label-counts',
            desc='The section label counts across splits',
            df=df,
            meta=(('section', 'the target discharge summary section'),
                  ('train', 'the training split number'),
                  ('test', 'the test split number'),
                  ('validation', 'the validation split number')))

    @property
    def note_section_align_contingency(self) -> DataFrameDescriber:
        """Shows the data direction from source notes to summary sections."""
        df: pd.DataFrame = self.feature_dataframe
        df = df[~df['smy_sec'].isnull()]
        return self._create_contingency(
            df=df,
            left_col='src_cat',
            top_col='smy_sec',
            left_desc='source note',
            top_desc='summary section',
            name='note-section-aligns',
            desc='Note to section alignment counts')

    @property
    def informal_eval_questions(self) -> DataFrameDescriber:
        rows = (('preference', 'Do you prefer the generated summary'),
                ('readablility', 'Of the data in the generated summary, how readable is it'),
                ('correctness', 'Of the data that is in the generated summary, how correct is it'),
                ('complete', 'How complete is the generated summary'),
                ('sections', 'Of the data in the summary, how well is it sectioned'))
        return DataFrameDescriber(
            name='informal-eval-questions',
            desc='Informal evaluation questions answered by physicians',
            df=pd.DataFrame(rows, columns='desc question'.split()),
            meta=(('desc', 'Question description/category'),
                  ('question', 'Question text given to physician')))

    @property
    def mimic_admission_note_count_stats(self) -> DataFrameDescriber:
        cnts: Tuple[Tuple[int, int], ...] = \
            self.corpus.note_event_persister.get_note_counts()
        df = pd.DataFrame(cnts, columns='hadm_id count'.split())
        dfs = df['count'].describe().to_frame()
        dfs.columns = ['value']
        dfs.insert(0, 'description', dfs.index)
        dfs = dfs.reset_index(drop=True)
        return DataFrameDescriber(
            name='mimic3-admission-stats',
            desc='MIMIC-III note count by admission statistics',
            df=dfs)

    @property
    @persisted('_match_admission_note_counts')
    def match_admission_note_counts(self) -> DataFrameDescriber:
        dff: pd.DataFrame = self.feature_dataframe
        hadm_ids: List[int] = dff['hadm_id'].drop_duplicates().to_list()
        rows: List[Tuple[Any, ...]] = []
        hadm_id: int
        for hadm_id in hadm_ids:
            note_ids: Tuple[int, ...] = self.corpus.note_event_persister.\
                get_row_ids_by_hadm_id(hadm_id)
            rows.append((hadm_id, len(note_ids)))
        return DataFrameDescriber(
            name='match-admission-note-counts',
            desc='Match dataset note count by admission',
            df=pd.DataFrame(rows, columns='hadm_id count'.split()))

    @property
    def match_admission_note_count_stats(self) -> DataFrameDescriber:
        dfd = self.match_admission_note_counts
        dfs = dfd.df['count'].describe().to_frame()
        dfs.columns = ['value']
        dfs.insert(0, 'description', dfs.index)
        dfs = dfs.reset_index(drop=True)
        return DataFrameDescriber(
            name='match-dataset-admission-stats',
            desc='Match note count by admission statistics',
            df=dfs)

    @property
    def generated_stats(self) -> DataFrameDescriber:
        logger.info(f'reading: {self.human_eval_path}')
        gens: Set[int] = set(pd.read_excel(self.human_eval_path)['hadm_id'])
        msids: Set[int] = set(self.msid_anon_resource.note_ids['hadm_id'].astype(int).drop_duplicates())
        dsprovs: Set[int] = set(map(int, self.dsprov_stash.keys()))
        rows: List[Tuple[Any, ...]] = []
        rows.append(('dsprov', 'medsecid', len(dsprovs & msids)))
        rows.append(('generated', 'medsecid', len(gens & msids)))
        rows.append(('generated', 'dsprov', len(gens & dsprovs)))
        return DataFrameDescriber(
            name='generated-summary-stats',
            desc='Generated DS, medsecid and dsprov admission intersections',
            df=pd.DataFrame(rows, columns='d1 d2 count'.split()),
            meta=(('ds1', 'First dataset'),
                  ('ds2', 'Second dataset'),
                  ('count', 'Dataset intersection count')))

    def _get_data_describer(self) -> DataDescriber:
        return DataDescriber(
            name='dataset',
            describers=(
                self.informal_eval_questions,
                self.get_label_stats(),
                self.note_stats,
                self.graph_stats,
                self.note_section_align_contingency,
                self.informal_eval_questions,
                self.mimic_admission_note_count_stats,
                self.match_admission_note_count_stats,
                self.generated_stats))

    def save_figures(self):
        """Create a note to section alignment contingency figure."""
        dfd: DataFrameDescriber = self.note_section_align_contingency
        df: pd.DataFrame = dfd.df
        df.index.name = 'Source Note'
        df.columns.name = 'Discharge Summary Section'
        df.columns = list(map(
            lambda c: tw.fill(c, width=10, break_long_words=False),
            df.columns))
        self.contingency_table.create(
            name=HeatMapPlot,
            format='d',
            title=dfd.desc,
            dataframe=df,
            x_label_rotation=50,
            params={'cbar': False})
        self.contingency_table.save()


@dataclass
class ReportApplication(object):
    """A reporting application that creates the Zensols Latex (``datdesc``
    utility) system to generate tables and figures.

    """
    reporters: Dict[str, PerformanceReporter] = field()
    """The reporters by name, which is typically the kind of reporting it does.

    """
    base_dir: Path = field(default=Path('.'))
    """The base directory to create the CSV and YML ``datdesc`` configuration
    files.

    """
    def _save(self):
        reporter: PerformanceReporter
        for reporter in self.reporters.values():
            dd: DataDescriber = reporter.data_describer
            dd.save(
                output_dir=self.base_dir / 'config' / f'csv',#-{dd.name}',
                yaml_dir=self.base_dir / 'target' / f'table-raw',#-{dd.name}',
                include_excel=True)
            reporter.save_figures()

    def report(self):
        """Create report output."""
        self._save()
