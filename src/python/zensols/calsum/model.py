"""Model and domain classes.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Set, Dict, Sequence
from dataclasses import dataclass, field
import logging
import collections
from itertools import chain
from io import StringIO
from pathlib import Path
from zensols.util import Failure
from zensols.persist import persisted, Stash
from zensols.nlp import (
    LexicalSpan, FeatureToken, FeatureSentence, FeatureDocument,
    FeatureDocumentParser
)
from zensols.deeplearn.batch import DataPoint
from zensols.deeplearn.result import ResultsContainer
from zensols.deeplearn.model import ModelFacade
from zensols.deepnlp.classify import ClassificationPredictionMapper
from zensols.deepnlp.transformer import WordPieceFeatureDocumentFactory
from zensols.mimic import (
    NoteEvent, NoteEventPersister, Section, Note, HospitalAdmission, Corpus
)
from zensols.mimic.regexnote import DischargeSummaryNote
from zensols.mimicsid import PredictedNote
from .matcher import SentenceMatch, SentenceMatchSet

logger = logging.getLogger(__name__)


@dataclass
class SentenceMatchSetDataPoint(DataPoint):
    """The model's batched data point, which is a sentence pairing.

    """
    sent_match_set: SentenceMatchSet = field()
    """A set of matched source/summary sentences for an admission."""

    @property
    def labels(self) -> Tuple[str, ...]:
        """The discharge summary sections."""
        return self.sent_match_set.summary_sections

    @property
    def section_ids(self) -> Tuple[str, ...]:
        """The source note antecedent section names."""
        return self.sent_match_set.source_sections

    @property
    def note_categories(self) -> Tuple[str, ...]:
        """The source note antecedent categories (i.e. ``radiology``)."""
        return self.sent_match_set.note_categories

    @property
    @persisted('_doc', transient=True)
    def doc(self) -> FeatureDocument:
        """The source note antecedent document.  All sentences from all note
        antecedents are added to this document.  The number of sentences is the
        same as :obj:`labels`, :obj:`section_ids`, and :obj:`note_categories`.

        """
        return FeatureDocument(
            sents=tuple(map(lambda m: m.source_sent,
                            self.sent_match_set.matches)))


@dataclass
class SequencePredictionMapper(ClassificationPredictionMapper):
    """Predicts which sentences to add to the discharge summary and in what
    section (if any).

    """
    doc_parser: FeatureDocumentParser = field(default=None)
    """The document parsed used to create a one sentence document when nothing
    is selected.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)

    """The feature document factory that populates embeddings."""

    keep_notes: Set[str] = field(default=None)
    """The note antecedents by category to add to the discharge summary."""

    keep_summary_sections: Sequence[str] = field(default=None)
    """The discharge summary sections add to the discharge summary in order."""

    default_note: str = field(default=None)
    """The note cateogry to use for empty notes."""

    def __post_init__(self):
        super().__post_init__()
        h2n = Note.category_to_id
        self.keep_notes = frozenset(map(h2n, self.keep_notes))
        self.default_note = h2n(self.default_note)

    def _add_embedding(self, doc: FeatureDocument):
        """Add embeddings to document (only sentinal is used)."""
        if self.word_piece_doc_factory is not None:
            self.word_piece_doc_factory.populate(doc, True)

    def _create_match_set(self, adm: HospitalAdmission) -> SentenceMatchSet:
        """Create a sentence match data structure consumable by the model."""
        matches: List[SentenceMatch] = []
        note: Note
        for note in adm.notes:
            cat: str = Note.category_to_id(note.category)
            if cat not in self.keep_notes or \
               note.category == DischargeSummaryNote.CATEGORY:
                # skip the discharge summaries of test data and any other note
                # not specified by the physicians to keep
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'skipping note {cat}')
                continue
            # add sentences of paragraphs of sections
            sec: Section
            for sec in note.sections.values():
                para: FeatureDocument
                for para in sec.paragraphs:
                    self._add_embedding(para)
                    sent: FeatureSentence
                    for sent in para:
                        sm = SentenceMatch()
                        sm.source_sent = sent
                        sm.source_note_category = cat
                        sm.source_section = sec.name
                        matches.append(sm)
        return SentenceMatchSet(
            hadm_id=adm.hadm_id,
            matches=tuple(matches))

    def _add_newlines(self, sents: List[FeatureSentence], n: int):
        """Add ``n`` newlines tokens to the last sentence of ``sents``."""
        last: FeatureSentence = sents[-1]
        lt: FeatureToken = last.tokens[-1]
        idx: int = lt.idx + len(lt.norm)
        tok = FeatureToken(
            i=lt.i + 1,
            idx=idx,
            i_sent=len(last) + 1,
            norm=('\n' * n),
            lexspan=LexicalSpan(idx, idx + n))
        # add features needed to reindex later
        tok.is_space = True
        tok.is_punctuation = False
        tok.sent_i = len(sents) - 1
        # detach to generate other expected features
        tok = tok.detach(skip_missing=True)
        last.tokens = (*last.tokens, tok)

    def _index_matches(self, secs: List[List[str]], mset: SentenceMatchSet) -> \
            List[Tuple[FeatureSentence, List[FeatureSentence]]]:
        """Return sections as a tuples of ``(<header sentence>, <body
        sentences>)``.

        """
        ds_secs: Dict[str, Section] = collections.defaultdict(list)
        sec_sents: List[Tuple[FeatureSentence, List[FeatureSentence]]] = []
        assert len(secs) == len(mset.matches)
        sec: str
        sm: SentenceMatch
        for sec, sm in zip(secs, mset.matches):
            if sec != FeatureToken.NONE:
                ds_secs[sec].append(sm.source_sent)
        for sec in self.keep_summary_sections:
            sents: List[FeatureSentence] = ds_secs.get(sec)
            if sents is not None:
                header: str = f'{Section.name_to_header(sec)}:\n'
                header_doc: FeatureDocument = self.doc_parser(header)
                self._add_newlines(header_doc.sents, 1)
                self._add_newlines(sents, 2)
                for sent in sents:
                    sent.clear()
                    sent.text = sent.norm
                sents[-1].text = sents[-1].text + '\n\n'
                header_doc[0].text = header_doc[0].norm + '\n'
                sec_sents.append((sec, header_doc[0], sents))
        # the model might not have classified any sections
        if len(sec_sents) > 0:
            # remove trailing two newlines from the last sentence
            sec_sents[-1][2][-1].strip()
        else:
            # if no matches were found, create a document with what amounts to
            # an error message
            sec_name: str = Note.DEFAULT_SECTION_NAME
            header_doc: FeatureDocument = self.doc_parser(
                f'{Section.name_to_header(sec_name)}:\n')
            body_doc: FeatureDocument = self.doc_parser('No data')
            self._add_newlines(header_doc.sents, 1)
            self._add_embedding(header_doc)
            self._add_embedding(body_doc)
            sec_sents.append((sec_name, header_doc, body_doc))
        return sec_sents

    def _create_summary(self, secs: List[List[str]],
                        mset: SentenceMatchSet) -> Note:
        """Create the discharge summary note."""
        note_text = StringIO()
        header_spans: List[LexicalSpan] = []
        body_spans: List[LexicalSpan] = []
        body_sents: List[FeatureSentence] = []
        start: int = 0
        sec_sents: List[Tuple[FeatureSentence, List[FeatureSentence]]] = \
            self._index_matches(secs, mset)
        ix: int
        sec_name: str
        header: FeatureSentence
        body: List[FeatureSentence]
        for ix, (sec_name, header, body) in enumerate(sec_sents):
            start: int = len(note_text.getvalue())
            # add header text to the note text
            note_text.write(header.text)
            # remove the colon and newline from the header span
            header_spans.append(LexicalSpan(
                start, len(note_text.getvalue()) - 2))
            # add the sentence orth to the note text
            note_text.write(' '.join(map(lambda s: s.text, body)))
            # the body span includes the newlines that won't be stripped later
            nl: int = 2 if ix < len(sec_sents) - 1 else 0
            body_spans.append(LexicalSpan(
                start, len(note_text.getvalue()) - nl))
            # add the header and body feature sentences
            body_sents.append(header)
            body_sents.extend(body)
        # create the body document used for the note
        body_doc = FeatureDocument(
            text=note_text.getvalue(),
            sents=tuple(body_sents))
        # make indexes for all sentences consistent in the context of the
        # document
        body_doc.reindex()
        # add sections
        psecs: List[Section] = []
        sdata = zip(header_spans, body_spans, sec_sents)
        for sid, (hspan, bspan, (sec_name, header, body)) in enumerate(sdata):
            psecs.append(Section(
                id=sid,
                name=sec_name,
                container=body_doc,
                header_spans=(hspan,),
                body_span=bspan))
        # create the note with the document and sections
        return PredictedNote(
            predicted_sections=tuple(psecs),
            doc=body_doc)

    def _create_features(self, adm: HospitalAdmission) -> \
            Tuple[FeatureSentence, ...]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating features for {adm}')
        mset: SentenceMatchSet = self._create_match_set(adm)
        self._docs.append(mset)
        return (mset,)

    def map_results(self, result: ResultsContainer) -> \
            Tuple[Tuple[SentenceMatchSet, Note], ...]:
        sec_set: List[List[str]] = self._map_classes(result)
        assert len(sec_set) == len(self._docs)
        return tuple(map(lambda t: (t[1], self._create_summary(*t)),
                         zip(sec_set, self._docs)))


@dataclass
class DischargeSummaryGenerator(object):
    """Generates discharge summaries using the model.

    """
    corpus: Corpus = field()
    """The MIMIC-III corpus."""

    facade: ModelFacade = field()
    """The model facade for use with the prediction API."""

    keep_notes: Set[str] = field()
    """The note antecedents by category to add to the discharge summary."""

    output_dir: Path = field()
    """The output directory."""

    def _process_admission(self, hadm_id: str):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'generating summary for admission: {hadm_id}')

        adm: HospitalAdmission = self.corpus.get_hospital_adm_by_id(hadm_id)
        pred_note: Note = self.facade.predict([adm])[0][1]
        adm_path: Path = self.output_dir / hadm_id
        np: NoteEventPersister = self.corpus.note_event_persister
        by_cat: Dict[str, List[int]] = np.get_row_ids_by_category(
            int(hadm_id), self.keep_notes)
        row_ids: Tuple[str, ...] = tuple(
            chain.from_iterable(map(lambda t: tuple(map(str, t[1])),
                                    by_cat.items())))
        for row_id in row_ids:
            note: NoteEvent = np.get_by_id(row_id)
            if note.category == DischargeSummaryNote.CATEGORY:
                note_path: Path = adm_path / f'{row_id}-discharge-summary-orignial.txt'
                note_path.parent.mkdir(parents=True, exist_ok=True)
                with open(note_path, 'w') as f:
                    f.write(note.text)
                continue
            cat: str = Note.category_to_id(note.category)
            note_path: Path = adm_path / f'{row_id}-{cat}.txt'
            note_path.parent.mkdir(parents=True, exist_ok=True)
            with open(note_path, 'w') as f:
                f.write(note.text)
        pred_note_path: Path = adm_path / 'discharge-summary.txt'
        pred_note_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_note_path, 'w') as f:
            f.write(pred_note.text)
        pred_note_path: Path = adm_path / 'discharge-summary-sectioned.txt'
        pred_note_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_note_path, 'w') as f:
            pred_note.write(writer=f)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote admission: {adm_path}')

    def generate(self, hadm_ids: Tuple[str, ...] = None) -> List[str]:
        """Generate a discharge summary for admission ID ``hadm_id``."""
        if hadm_ids is None:
            from itertools import islice as isl
            feat_stash: Stash = self.facade.feature_stash
            test_stash: Stash = feat_stash.splits['test']
            hadm_ids = isl(test_stash.keys(), 10)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'generating {len(test_stash)} summaries')
        success_hadm_ids: List[str] = []
        hadm_id: str
        for hadm_id in test_stash.keys():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'processing admission: {hadm_id}')
            try:
                self._process_admission(hadm_id)
                success_hadm_ids.append(hadm_id)
            except Exception as e:
                failure = Failure(
                    exception=e,
                    message=f'Could not generate admission: {hadm_id}')
                failure.write()
        return success_hadm_ids
