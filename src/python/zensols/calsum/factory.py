"""Create clinical AMR graphs.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Iterable
from dataclasses import dataclass, field
import sys
import itertools as it
from igraph import Graph
from zensols.amr import AmrDocument, AmrFeatureSentence, AmrFeatureDocument
from zensols.clinicamr import (
    NoteDocument, SectionDocument, AdmissionAmrFeatureDocument
)
from zensols.calamr import (
    DocumentNode, AmrDocumentNode, DocumentGraphComponent, DocumentGraphFactory
)
from zensols.calamr.summary.factory import SummaryConstants


@dataclass
class AdmissionAmrDocumentGraphFactory(DocumentGraphFactory):
    """Creates graphs that use the structure of
    :class:`.AdmissionAmrFeatureDocument` clinical notes.

    """
    para_limit: int = field(default=sys.maxsize)
    """The max number of paragraphs to add to the graph."""

    sec_limit: int = field(default=sys.maxsize)
    """The max number of sections to add to the graph."""

    note_limit: int = field(default=sys.maxsize)
    """The max number of notes to add to the graph."""

    def _create_section(self, root: AdmissionAmrFeatureDocument,
                        sec: SectionDocument) -> DocumentNode:
        """Create a clinical section document node (connects to paragraphs below
        and section document nodes higher closer to the root).

        """
        paras: List[AmrDocumentNode] = []
        para_docs: Iterable[AmrFeatureDocument] = it.islice(
            sec.create_paragraphs(), self.para_limit)
        para: AmrFeatureDocument
        for pix, para in enumerate(para_docs):
            paras.append(AmrDocumentNode(
                context=self.graph_attrib_context,
                name=f'para {pix}',
                root=root,
                children=(),
                doc=para))
        return DocumentNode(
            context=self.graph_attrib_context,
            name=sec.name,
            root=root,
            children=tuple(paras))

    def _create_note(self, root: AdmissionAmrFeatureDocument,
                     note: NoteDocument) -> DocumentNode:
        """Create a clinical note document node (connects to source root)."""
        childs: List[DocumentNode] = []
        sec: SectionDocument
        for sec in it.islice(note.create_sections(), self.sec_limit):
            childs.append(self._create_section(root, sec))
        return DocumentNode(
            context=self.graph_attrib_context,
            name=note.category,
            root=root,
            children=tuple(childs))

    def _create_node_root(self, root: AdmissionAmrFeatureDocument,
                          notes: Iterable[NoteDocument]) -> \
            AmrFeatureDocument:
        """Create a document that will be used as the
        :obj:`.AmrDocumentNode.doc`.

        """
        sents: List[AmrFeatureSentence] = []
        note: NoteDocument
        for note in it.islice(notes, self.note_limit):
            sec: SectionDocument
            for sec in it.islice(note.create_sections(), self.sec_limit):
                para: AmrFeatureDocument
                for para in it.islice(sec.create_paragraphs(), self.para_limit):
                    sents.extend(para.sents)
        return root.from_sentences(sents)

    def _create_source(self, root: AdmissionAmrFeatureDocument) -> \
            AmrDocumentNode:
        """Create the source root node and component of the bipartite graph from
        the note antecedents.

        """
        childs: List[DocumentNode] = []
        note: NoteDocument
        for note in it.islice(root.create_note_antecedents(), self.sec_limit):
            childs.append(self._create_note(root, note))
        return AmrDocumentNode(
            context=self.graph_attrib_context,
            name=SummaryConstants.SOURCE_COMP,
            root=root,
            children=tuple(childs),
            doc=self._create_node_root(
                root, root.create_note_antecedents()))

    def _create_summary(self, root: AdmissionAmrFeatureDocument) -> \
            AmrDocumentNode:
        """Create the summary root node and component of the bipartite graph
        from the discharge summary note .

        """
        node: DocumentNode = self._create_note(
            root, root.create_discharge_summary())
        return AmrDocumentNode(
            context=self.graph_attrib_context,
            name=SummaryConstants.SUMMARY_COMP,
            root=root,
            children=tuple(node,),
            doc=self._create_node_root(
                root, (root.create_discharge_summary(),)))

    def _create(self, root: AmrFeatureDocument) -> \
            Tuple[DocumentGraphComponent, ...]:
        def map_comp(node: DocumentNode) -> DocumentGraphComponent:
            graph: Graph = DocumentGraphComponent.graph_instance()
            return DocumentGraphComponent(graph, node)

        assert isinstance(root, AdmissionAmrFeatureDocument)
        assert isinstance(root.amr, AmrDocument)
        source: DocumentNode = self._create_source(root)
        summary: DocumentNode = self._create_summary(root)
        children = tuple(filter(lambda dl: len(dl) > 0, (source, summary)))
        return tuple(map(map_comp, children))
