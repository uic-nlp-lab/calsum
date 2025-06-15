"""Classes for extractive summary sentence node matching via bipartite graph
alignments.

"""
__author__ = 'Paul Landes'

from typing import (
    Tuple, List, Dict, Set, Iterable, Sequence, Union, Callable, Any, ClassVar
)
from dataclasses import dataclass, field
import logging
import sys
import gc
import textwrap as tw
import collections
from itertools import chain
from io import TextIOBase
from frozendict import frozendict
import numpy as np
import pandas as pd
from igraph import Graph, Vertex, Edge
from zensols.nlp import FeatureToken, FeatureSentence
from zensols.util import Failure
from zensols.persist import (
    persisted, Stash, ReadOnlyStash, PrimeableStash,
    PersistableContainer, NotPickleable,
)
from zensols.config import Dictable, Writable
from zensols.amr import AmrFeatureDocument
from zensols.calamr import (
    GraphEdge, ComponentAlignmentGraphEdge,
    GraphNode, SentenceGraphNode, DocumentGraphNode,
    DocumentGraph, GraphComponent
)
from zensols.alsum.graph import ReducedGraph
from zensols.calamr.summary.factory import SummaryConstants
from . import CalsumError

logger = logging.getLogger(__name__)


@dataclass
class _SummarySent(Dictable):
    """A sentence graph vertex/node in the summary component of the graph.

    """
    vertex: Vertex = field(repr=False)
    """Vertex in the graph."""

    parent_edge: Edge = field(repr=False)
    """The parent edge between the sentence and paragraph node."""

    section: str = field()
    """The section the sentence is a member of."""

    section_index: int = field()
    """The position of the sentence in the section."""

    @property
    def paragraph_index(self) -> int:
        """The index of the sentence in the contained paragraph."""
        return self.node.sent_ix

    @property
    def node(self) -> SentenceGraphNode:
        """The node from the vertex."""
        return GraphComponent.to_node(self.vertex)

    @property
    def id(self) -> int:
        """The graph node ID."""
        return self.node.id

    @property
    def parent_graph_edge(self) -> GraphEdge:
        return GraphComponent.to_edge(self.parent_edge)

    @property
    def sent(self) -> AmrFeatureDocument:
        """The sentence from the sentence node."""
        return self.node.sent

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        text: str = tw.shorten(self.node.sent.text, 70)
        dct = self.asdict()
        dct['text'] = text
        self._write_dict(dct, depth, writer)

    def __str__(self) -> str:
        ix: int = self.section_index
        sec: str = self.section
        sent: str = tw.shorten(self.sent.text, 50)
        return f'{self.id}: index={ix}, sec={sec}: {sent}'


@dataclass
class _SourceAlign(Dictable):
    """An alignment from the source to the summary.  These are added to source
    component's nodes and latter aggregated into :class:`._SourceAlignSet`.
    Each alignment "traces" back to a sentence in the summary (:obj:`sent`).

    """
    sent: _SummarySent = field()
    """The summary sentence that contains the node connected by this alignment.

    """
    edge: ComponentAlignmentGraphEdge = field()
    """The alignment edge."""

    @property
    def summary_sent_flow(self) -> float:
        """The summary sentence flow."""
        return self.sent.parent_graph_edge.flow

    @property
    def edge_flow(self) -> float:
        """The source flow via the alignment edge."""
        return self.edge.flow

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_object(self.sent, depth, writer)
        self._write_line(f'sentence flow: {self.sent_flow}', depth, writer)
        self._write_line(f'align flow: {self.edge.flow}', depth, writer)


@dataclass
class _SourceAlignSet(Dictable, NotPickleable):
    """A grouping alignments added to the source component.  This represents all
    alignments for source sentence that are aligned with a summary sentence
    during the process of iterating through the summary.  However, after that,
    it represents the alignments per sentence in
    :obj:`_SourceSent.aligns_by_id`.

    """
    aligns: Sequence[_SourceAlign] = field(default_factory=list)
    """The (source to summary) alignments that make up this grouping."""

    @property
    def summary_sent(self) -> _SummarySent:
        """The summary sentence.  There is only once since all alignments
        originate from the same summary sentence.

        """
        return self.aligns[0].sent

    @property
    def summary_sent_flow(self) -> float:
        """The summary sentence flow.  There is only once since all alignments
        originate from the same summary sentence.

        """
        return self.aligns[0].summary_sent_flow

    @property
    @persisted('_align_flows')
    def align_flows(self) -> np.ndarray:
        """The source alignment flows."""
        return np.array(tuple(map(lambda sa: sa.edge_flow, self.aligns)))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        sent: _SummarySent = self.aligns[0].sent
        self._write_object(sent, depth, writer)
        self._write_line(f'sentence flow: {self.summary_sent_flow}',
                         depth, writer)
        self._write_line(f'align flows: {self.align_flows}', depth, writer)

    def __str__(self) -> str:
        smy_sent: _SummarySent = self.summary_sent
        sentt: str = smy_sent.node.sent.norm
        ix: int = smy_sent.section_index
        sec: str = smy_sent.section
        sm: float = self.align_flows.sum()
        sf: float = self.summary_sent_flow
        id: int = smy_sent.id
        return (f'{id}: index={ix}, af={sm:.3f}, smyf={sf:.3f}: ' +
                f'sec={sec}: {sentt}')


@dataclass
class _SourceSent(Dictable, NotPickleable):
    """A sentence graph vertex/node in the source component of the graph.  It
    contains the alignments to the summary sentence(s).

    """
    _DICTABLE_ATTRIBUTES = {'id'}
    _DICTABLE_WRITABLE_DESCENDANTS = True

    vertex: Vertex = field(repr=False)
    """Vertex in the graph."""

    parent_edge: Edge = field(repr=False)
    """The parent edge between the sentence and paragraph node."""

    aligns: Tuple[_SourceAlign, ...] = field(repr=False)
    """A grouping of alignments added to the source component found in
    :obj:`sent`.

    """
    section: str = field()
    """The section the sentence is a member of."""

    note_category: str = field()
    """The note cateogory"""

    @property
    def node(self) -> SentenceGraphNode:
        """The sourcenode from the vertex."""
        return GraphComponent.to_node(self.vertex)

    @property
    def id(self) -> int:
        """The graph node ID."""
        return self.node.id

    @property
    def parent_graph_edge(self) -> GraphEdge:
        return GraphComponent.to_edge(self.parent_edge)

    @property
    def sent(self) -> AmrFeatureDocument:
        """The source sentence from the sentence node."""
        return self.node.sent

    @property
    @persisted('_aligns_by_id')
    def aligns_by_id(self) -> Dict[int, _SourceAlignSet]:
        """The summary to source alignments keyed by their summary sentence
        graph node IDs.

        """
        def map_als(aligns: List[_SourceAlign]) -> _SourceAlignSet:
            aligns: Tuple[_SourceAlign, ...] = tuple(aligns)
            assert len(aligns) > 0
            return _SourceAlignSet(aligns)

        aligns: Dict[str, List[_SourceAlign]] = collections.defaultdict(list)
        for align in self.aligns:
            aligns[align.sent.id].append(align)
        return frozendict(map(lambda t: (t[0], map_als(t[1])), aligns.items()))

    @property
    def source_sent_flow(self) -> float:
        """The source sentence flow."""
        return self.parent_graph_edge.flow

    @property
    @persisted('_align_flow_sums')
    def align_flow_sums(self) -> np.ndarray:
        """The (respective) summary to source alignment flows."""
        sass: Tuple[_SourceAlignSet] = tuple(self.aligns_by_id.values())
        return np.array(tuple(map(lambda a: a.align_flows.sum(), sass)))

    @property
    @persisted('_align_flow_sum')
    def align_flow_sum(self) -> np.ndarray:
        """The sum of all (respective) summary to source alignment flows."""
        return self.align_flow_sums.sum()

    @property
    @persisted('_align_flow_var')
    def align_flow_var(self) -> np.ndarray:
        """The variation of the (respective) summary to source alignment flows.

        """
        return self.align_flow_sums.var()

    def get_distribution(self, min_sent_flow: float = None) -> \
            Tuple[Tuple[float, _SourceAlignSet], ...]:
        """Get source alignments as tuples of the sum of their alignment flows
        between the summary sentence nodes and their aligned source nodes.

        :param min_sent_flow: the lowest amount of summary sentence flow;
                              otherwise the alignment set is omitted

        :return: tuples of ``(<total align flow>, <alignment set>)``

        """
        sass: Iterable[_SourceAlignSet] = self.aligns_by_id.values()
        if min_sent_flow is not None:
            sass = filter(
                lambda a: a.summary_sent_flow >= min_sent_flow, sass)
        sass = tuple(sass)
        aflows = np.array(tuple(map(lambda a: a.align_flows.sum(), sass)))
        norm: np.ndarray = aflows / aflows.sum()
        return tuple(sorted(zip(norm, sass), key=lambda t: t[0], reverse=True))

    @property
    @persisted('_distribution')
    def distribution(self) -> Tuple[Tuple[float, _SourceAlignSet], ...]:
        """See :mth:`get_distribution`."""
        return self.get_distribution()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_aligns: bool = False, include_dist: bool = True,
              min_sent_flow: float = None):
        text: str = tw.shorten(self.node.sent.text, 120)
        dct = self.asdict()
        self._write_dict(dct, depth, writer)
        self._write_line(f'text: {text}', depth, writer)
        self._write_line(f'souce_sent_flow: {self.source_sent_flow}',
                         depth, writer)
        self._write_line(f'align_flow_sum: {self.align_flow_sum:.3f}',
                         depth, writer)
        self._write_line(f'align_flow_var: {self.align_flow_sums.var():.3f}',
                         depth, writer)
        self._write_line(f'align_flow_stddev: {self.align_flow_sums.std():.3f}',
                         depth, writer)
        if include_aligns:
            self._write_line('summary:', depth, writer)
            self._write_object(self.aligns_by_id, depth + 1, writer)
        if include_dist:
            self._write_line('summary_distribution:', depth, writer)
            if min_sent_flow is None:
                dist = self.distribution
            else:
                dist = self.get_distribution(min_sent_flow=min_sent_flow)
            for aflow, sas in dist:
                self._write_line(f'{aflow:.3f}: {sas}', depth + 1, writer)

    def __str__(self) -> str:
        return f'{self.sent}: note/sec={self.note_category}/{self.section}'


@dataclass
class _SentenceSet(Dictable):
    """A container for all sentence matching (via alignments) data."""
    source_sents: Tuple[_SourceSent, ...] = field()

    def get_distributions(self, min_sent_flow: float) -> \
            List[Tuple[Tuple[Tuple[float, _SourceAlignSet], ...], _SourceSent]]:
        def sort_key(t):
            afs = t[1].align_flow_sum
            return afs

        dists = tuple(map(lambda ss: (ss.get_distribution(min_sent_flow), ss),
                          self.source_sents))
        dists = filter(lambda t: len(t[0]) > 0, dists)
        return sorted(dists, key=sort_key, reverse=True)


@dataclass(init=False)
class SentenceMatch(PersistableContainer, Writable):
    """A match between a source and summary sentence.

    """
    source_sent: FeatureSentence = field(default=None)
    """The matched source sentence."""

    source_note_category: str = field(default=None)
    """The matched source sentence's containing note category."""

    source_section: str = field(default=None)
    """The matched source sentence's containing section."""

    summary_sent: FeatureSentence = field(default=None)
    """The matched summary sentence."""

    summary_section: str = field(default=None)
    """The matched summary sentence's containing section."""

    summary_section_index: str = field(default=None)
    """The position of the summary sentence in the containing section."""

    def __init__(self, source: _SourceSent = None,
                 summary: _SummarySent = None):
        if source is not None:
            self.source_sent = source.sent
            self.source_note_category = source.note_category
            self.source_section = source.section
        if summary is not None:
            self.summary_sent = summary.sent
            self.summary_section = summary.section
            self.summary_section_index = summary.section_index

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('source:', depth, writer)
        self._write_line(f'sent: {self.source_sent}', depth + 1, writer)
        self._write_line(f'note category: {self.source_note_category}',
                         depth + 1, writer)
        self._write_line(f'section: {self.source_section}', depth + 1, writer)
        self._write_line('summary:', depth, writer)
        self._write_line(f'sent: {self.summary_sent}', depth + 1, writer)
        self._write_line(f'section: {self.summary_section}', depth + 1, writer)
        self._write_line(f'section_index: {self.summary_section_index}',
                         depth + 1, writer)

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'source_sent'):
            self._try_deallocate(self.source_sent)
        self.source_sent = None
        if hasattr(self, 'summary_sent'):
            self._try_deallocate(self.summary_sent)
        self.summary_sent = None


@dataclass
class SentenceMatchSet(PersistableContainer, Writable):
    """A set of matched source/summary sentences for an admission.

    """
    hadm_id: str = field()
    """The admission ID."""

    matches: Tuple[SentenceMatch, ...] = field()
    """The setence matches for the admission."""

    def __post_init__(self):
        super().__init__()

    def _map_label(self, lab: str) -> str:
        if lab is None:
            return FeatureToken.NONE
        else:
            return lab.replace(' ', '-')

    @property
    @persisted('_smy_sections', transient=True)
    def summary_sections(self) -> Tuple[str, ...]:
        return tuple(map(lambda m: self._map_label(m.summary_section),
                         self.matches))

    @property
    @persisted('_src_sections', transient=True)
    def source_sections(self) -> Tuple[str, ...]:
        return tuple(map(lambda m: self._map_label(m.source_section),
                         self.matches))

    @property
    @persisted('_note_categories', transient=True)
    def note_categories(self) -> Tuple[str, ...]:
        return tuple(map(lambda m: self._map_label(m.source_note_category),
                         self.matches))

    @property
    def dataframe(self) -> pd.DataFrame:
        rows: Tuple[Any, ...] = []
        for sm in self.matches:
            rows.append((
                sm.source_note_category, sm.source_section,
                sm.summary_section, sm.summary_section_index))
        return pd.DataFrame(rows, columns=(
            'src_cat src_sec smy_sec smy_sec_ix'.split()))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'admission: {self.hadm_id}', depth, writer)
        smatch: SentenceMatch
        for i, smatch in enumerate(self.matches):
            if i > 0:
                self._write_divider(depth, writer)
            self._write_object(smatch, depth, writer)

    def deallocate(self):
        super().deallocate()
        for m in self.matches:
            self._try_deallocate(m)
        del self.matches

    def __str__(self) -> str:
        return f'{self.hadm_id}: matches: {len(self.matches)}'


@dataclass
class _SentenceMatcher(object):
    """Matches source to summary sentences by creating using the reduced graph
    alignments.

    """
    min_sent_flow: float = field()
    """Minimum flow needed to add matches in
    :meth:`._SentenceSet.get_distributions`.

    """
    def _get_doc_node_parent(self, c: Vertex) -> Vertex:
        """Get a document node parent (closer to root) of child node ``c``."""
        to_node: Callable = GraphComponent.to_node
        nvs: List[Vertex] = c.neighbors('in')
        ngns: Tuple[GraphNode, ...] = tuple(filter(
            lambda t: isinstance(t[1], DocumentGraphNode),
            map(lambda v: (v, to_node(v)), nvs)))
        if len(ngns) != 1:
            raise CalsumError(
                f'Expecting singleton parent doc node but got: {ngns}')
        return ngns[0][0]

    def _get_alignments(self, graph: ReducedGraph) -> Dict[int, Set[Edge]]:
        """Return alignments keyed by target summary component vertex ID."""
        dg: DocumentGraph = graph.doc_graph
        cedges: Dict[int, Set[Edge]] = collections.defaultdict(set)
        src_comp: DocumentGraph = \
            dg.components_by_name[SummaryConstants.SOURCE_COMP]
        smy_comp: DocumentGraph = \
            dg.components_by_name[SummaryConstants.SUMMARY_COMP]
        e: Edge
        ge: GraphEdge
        for e, ge in dg.es.items():
            if isinstance(ge, ComponentAlignmentGraphEdge):
                v: Vertex = dg.vertex_ref_by_id(e.target)
                assert v in smy_comp.vs
                assert v not in src_comp.vs
                if logger.isEnabledFor(logging.TRACE):
                    gn: GraphNode = dg.to_node(v)
                    logger.trace(f'adding {gn} ({gn.id}) ({e.target})')
                cedges[e.target].add(e)
        return cedges

    def _populate_summary_nodes(self, graph: ReducedGraph,
                                cedges: Dict[int, Set[Edge]],
                                smy_sent: _SummarySent):
        """Add the summary sentence ``ss`` to source graph aligned nodes."""
        from .attr import ClinicalConceptGraphNode
        dg: DocumentGraph = graph.doc_graph
        g: Graph = dg.graph
        src_comp: DocumentGraph = \
            dg.components_by_name[SummaryConstants.SOURCE_COMP]
        smy_comp: DocumentGraph = \
            dg.components_by_name[SummaryConstants.SUMMARY_COMP]
        # depth first search from the summary sentence
        smy_v: Vertex
        for smy_v in g.vs[g.dfs(smy_sent.vertex.index, mode='out')[0]]:
            # comp alignment vertex
            ces: List[Edge] = cedges.get(smy_v.index, ())
            if logger.isEnabledFor(logging.TRACE) and len(ces) > 0:
                ids: str = ', '.join(map(
                    lambda e: f'{dg.node_by_id(e.source).id}', ces))
                logger.trace(f'{dg.to_node(smy_v).id} -> {ids}')
            ce: Edge
            for ce in ces:
                # get the alignment graph attr
                cage: ComponentAlignmentGraphEdge = dg.to_edge(ce)
                # connect to the summary (from/source) node via the alignment
                src_v: Vertex = g.vs[ce.source]
                src: GraphNode = dg.to_node(src_v)
                if isinstance(src, ClinicalConceptGraphNode):
                    print(type(src), src, '_' * 20)
                # sanity check on data flow direction
                assert src_v in src_comp.vs and src_v not in smy_comp.vs
                assert smy_v in smy_comp.vs and smy_v not in src_comp.vs
                if logger.isEnabledFor(logging.TRACE):
                    # source (to/target) node and alignment flow
                    smy: GraphNode = dg.to_node(smy_v)
                    flow: float = cage.flow
                    logger.trace(f'{smy}->{src} ({src.attrib_type}): f={flow}')
                # add the summary sentence data
                src_align_set: _SourceAlignSet
                if hasattr(src, 'src_align_set'):
                    src_align_set = src.src_align_set
                else:
                    src_align_set = _SourceAlignSet()
                    src.src_align_set = src_align_set
                src_align_set.aligns.append(_SourceAlign(smy_sent, cage))

    def _populate_source_sents(self, graph: ReducedGraph,
                               cedges: Dict[int, Set[Edge]]):
        """Add summary sentinal data in component aligned source nodes."""
        doc_graph: DocumentGraph = graph.doc_graph
        g: Graph = doc_graph.graph
        comp: DocumentGraph = doc_graph.\
            components_by_name[SummaryConstants.SUMMARY_COMP]
        # nodes are created in the doc's sentence order
        prev_sec: str = None
        for vix in g.neighborhood(comp.root, 3, 'out', 3):
            # get the summary sentence's vertex
            sv: Vertex = comp.vertex_ref_by_id(vix)
            # the parent edge between the sentence and paragraph node
            se: Edge = sv.incident('in')[0]
            # paragraph up one level from sentences
            para_v: Vertex = self._get_doc_node_parent(sv)
            # sections are one level up from paragraphs
            sec_v: Vertex = self._get_doc_node_parent(para_v)
            sec: DocumentGraphNode = comp.to_node(sec_v)
            # increment sentence indicies when sections change
            if sec.label != prev_sec:
                six = 0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{comp.to_node(sv)}: section={sec.label}')
            # add this summary sentence's info in source nodes
            self._populate_summary_nodes(
                graph, cedges, _SummarySent(sv, se, sec.label, six))
            prev_sec, six = sec.label, six + 1

    def _get_source_align_sets(self, graph: ReducedGraph, sent_vert: Vertex) \
            -> Iterable[_SourceAlignSet]:
        """Get source aligned sets for subtree sentence node ``sent_vert``."""
        doc_graph: DocumentGraph = graph.doc_graph
        g: Graph = doc_graph.graph
        comp: DocumentGraph = doc_graph.\
            components_by_name[SummaryConstants.SOURCE_COMP]
        # depth first search, but avoiding alignments to summary component
        # (source graph only needed because source flows to the summary)
        src_vs: Iterable[Vertex] = filter(
            lambda v: v in comp.vs,
            g.vs[g.dfs(sent_vert.index, mode='out')[0]])
        # depth first search from the source sentence
        src_v: Vertex
        for src_v in src_vs:
            src: GraphNode = GraphComponent.to_node(src_v)
            if hasattr(src, 'src_align_set'):
                yield src.src_align_set

    def _get_source_sents(self, graph: ReducedGraph) -> _SentenceSet:
        """Bubble up and associate the summary sentence data added to the source
        graph's sentences.

        """
        source_sents: List[_SourceSent] = []
        doc_graph: DocumentGraph = graph.doc_graph
        g: Graph = doc_graph.graph
        comp: DocumentGraph = doc_graph.\
            components_by_name[SummaryConstants.SOURCE_COMP]
        # iterate over source sentence notes; nodes are created in the doc's
        # sentence order
        vix: int
        for vix in g.neighborhood(comp.root, 4, 'out', 4):
            # get the summary sentence's vertex
            sv: Vertex = comp.vertex_ref_by_id(vix)
            # the parent edge between the sentence and paragraph node
            se: Edge = sv.incident('in')[0]
            # paragraph up one level from sentences
            para_v: Vertex = self._get_doc_node_parent(sv)
            # sections are one level up from paragraphs
            sec_v: Vertex = self._get_doc_node_parent(para_v)
            # notes are one level up from sections
            note_v: Vertex = self._get_doc_node_parent(sec_v)
            # get source alignment sets
            asets: Tuple[_SourceAlignSet, ...] = \
                tuple(self._get_source_align_sets(graph, sv))
            if logger.isEnabledFor(logging.TRACE):
                for aset in asets:
                    aset.write_to_log(logger, logging.TRACE)
            sec: DocumentGraphNode = comp.to_node(sec_v)
            note: DocumentGraphNode = comp.to_node(note_v)
            if logger.isEnabledFor(logging.DEBUG):
                sgn: SentenceGraphNode = comp.to_node(sv)
                logger.debug(f'{sgn.sent.text}: note={note.label}, ' +
                             f'section={sec.label}')
            if len(asets) == 0:
                # ignore 0-flow sentences, which would be removed if pruning was
                # turned on in the reduced graph
                continue
            source_sents.append(_SourceSent(
                aligns=tuple(chain.from_iterable(
                    map(lambda s: s.aligns, asets))),
                vertex=sv,
                parent_edge=se,
                section=sec.label,
                note_category=note.label))
        return _SentenceSet(tuple(source_sents))

    def _map_sentences(self, sent_set: _SentenceSet) -> List[SentenceMatch]:
        """Map source to summary sentences.  Each source sentence is assigned
        summary sentence, or none at all.

        """
        matches: List[SentenceMatch] = []
        smy_matches: Set[int] = set()
        src_matches: Set[int] = set()
        dists = sent_set.get_distributions(self.min_sent_flow)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sents: {len(sent_set.source_sents)}, ' +
                         f'distributions: {len(dists)}')
        dist: Tuple[Tuple[float, _SourceAlignSet], ...]
        ss: _SourceSent
        for dist, ss in dists:
            aflow: float
            sas: _SourceAlignSet
            for aflow, sas in dist:
                smy_sent: _SummarySent = sas.summary_sent
                smy_sent_id: int = smy_sent.id
                if smy_sent_id in smy_matches:
                    continue
                matches.append(SentenceMatch(
                    source=ss,
                    summary=smy_sent))
                src_matches.add(ss.id)
                smy_matches.add(smy_sent_id)
                break
        for dist, ss in dists:
            matches.append(SentenceMatch(
                source=ss,
                summary=None))
        return matches

    def match(self, graph: ReducedGraph):
        """Summarize a corpus document.

        :param graph: the aligned reduced graph on which to match sentences

        """
        # create summary vertex alignment edge map
        cedges: Dict[int, Set[Edge]] = self._get_alignments(graph)
        # add summary sentinal data in component aligned source nodes
        self._populate_source_sents(graph, cedges)
        # bubble up and associate the summary sentence data added to the source
        # graph's sentences
        sent_set: _SentenceSet = self._get_source_sents(graph)
        # map source to summary sentences; each source sentence is assigned
        # summary sentence, or none at all
        matches: List[SentenceMatch] = self._map_sentences(sent_set)
        return SentenceMatchSet(hadm_id=None, matches=matches)


@dataclass
class SentenceMatchStash(ReadOnlyStash, PrimeableStash):
    """CRUDs instances of :class:`.SentenceMatchSet`.

    """
    _LOAD_ITER: ClassVar[int] = 0

    _sent_matcher: _SentenceMatcher = field()
    """The sentence matcher."""

    factory: Stash = field()
    """The stash that CRUDs :class:`~zensols.alsum.graph.ReducedGraph`."""

    def _maybe_gc(self):
        if (self.__class__._LOAD_ITER % 10) == 0:
            gc.collect()
        self.__class__._LOAD_ITER += 1

    def load(self, hadm_id: str) -> Union[Failure, SentenceMatchSet]:
        self.prime()
        self._maybe_gc()
        # igraph IDs will not match up since terminal nodes are
        # deleted; instead use graph attribute IDs
        graph: ReducedGraph = self.factory.get(hadm_id)
        if graph is not None:
            try:
                if graph.is_error:
                    # the graph couldn't be aligned
                    assert isinstance(graph.failure, Failure)
                    return graph.failure
                try:
                    match_set: SentenceMatchSet = \
                        self._sent_matcher.match(graph)
                    if len(match_set.matches) == 0:
                        raise CalsumError('No matches found for admission')
                except Exception as e:
                    # the aligned graph couldn't be matched
                    return Failure(
                        exception=e,
                        message=f'Could not sentence match {hadm_id}: {e}')
            finally:
                graph.deallocate()
            match_set.hadm_id = hadm_id
            return match_set

    def keys(self) -> Iterable[str]:
        self.prime()
        return self.factory.keys()

    def exists(self, hadm_id: str) -> bool:
        self.prime()
        return self.factory.exists(hadm_id)

    def prime(self):
        if isinstance(self.factory, PrimeableStash):
            self.factory.prime()

    def delete_alignment_errors(self):
        """Delete the aligned graph cache files that failed."""
        hadm_id: str
        graph: ReducedGraph
        for hadm_id, graph in self.factory.items():
            if graph.is_error:
                logger.info(f'deleting failure: {hadm_id}')
                self.factory.factory.delete(hadm_id)
