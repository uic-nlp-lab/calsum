"""Biomedical graph specialization.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
import logging
import re
from penman import Graph as PGraph
from penman import constant
from penman.graph import Instance, Attribute
from penman.surface import Alignment
from zensols.amr import AmrFeatureSentence
from zensols.calamr import AttributeGraphNode, ConceptGraphNode
from zensols.calamr.morph import IsomorphDocumentGraphDecorator
from .attr import ClinicalConceptGraphNode
from . import CalsumError, Cui

logger = logging.getLogger(__name__)


@dataclass
class ClinicalIsomorphDocumentGraphDecorator(IsomorphDocumentGraphDecorator):
    """Merges CUI information in notes that have CUI roles.  This adds the
    clinical concept information at the node that has the alignment with a
    entity linked token and then removes the child CUI role and node with which
    it is attached.

    :see: clinicamr resource library ``camr_token_cui_doc_decorator:name``

    """
    _CUI_REGEX = re.compile(r'^(C\d+):(.+)$')
    """Used to parse the CUI ID and the CUI description (preferred name)."""

    cui_role: str = field()
    """The string name of the CUI role annotated with a
    :class:`.zensols.amr.docparser.TokenAnnotationFeatureDocumentDecorator`.

    """
    def _get_cui(self, pg: PGraph, inst: Instance) -> Optional[Attribute]:
        attrs = tuple(filter(lambda a: a.role == self.cui_role,
                             pg.attributes(source=inst.source)))
        assert len(attrs) < 2
        if len(attrs) > 0:
            return attrs[0]

    def _parse_cui_attr(self, attr: Attribute) -> Tuple[str, str]:
        s: str = constant.evaluate(attr.target)
        m: re.Match = self._CUI_REGEX.match(s)
        if m is None:
            raise CalsumError(f'Attribute does not match to <CUIX:desc>: {s}')
        return m.groups()

    def _add_concepts(self, pg: PGraph, epis: Dict[Tuple[str, str, str], List],
                      concepts: List[ConceptGraphNode],
                      sent: AmrFeatureSentence, c_ixs: Dict[str, int]):
        ix: int
        inst: Instance
        for ix, inst in enumerate(pg.instances()):
            # merge CUI information in notes that have CUI roles
            epi: List = epis.get(inst)
            aligns = tuple(filter(lambda x: isinstance(x, Alignment), epi))
            params: Dict[str, Any] = dict(
                context=self._ctx.graph_attrib_context,
                triple=inst,
                token_aligns=aligns,
                sent=sent)
            cl: Type[ConceptGraphNode]
            cui_attr: Attribute = self._get_cui(pg, inst)
            if cui_attr is None:
                cl = ConceptGraphNode
            else:
                cl = ClinicalConceptGraphNode
                cui, pref_name = self._parse_cui_attr(cui_attr)
                params['cui'] = Cui(cui, pref_name)
            concepts.append(cl(**params))
            c_ixs[inst.source] = ix

    def _add_attributes(self, pg: PGraph,
                        epis: Dict[Tuple[str, str, str], List],
                        attribs: List[AttributeGraphNode],
                        sent: AmrFeatureSentence, c_ixs: Dict[str, int]):
        # add AMR attributes to add as igraph attributes later
        ix: int
        attr: Attribute
        for ix, attr in enumerate(pg.attributes()):
            epi: List = epis.get(attr)
            aligns = tuple(filter(lambda x: isinstance(x, Alignment), epi))
            if attr.role == self.cui_role:
                # don't add the roles since their information is added at the
                # concept level
                continue
            attribs.append(AttributeGraphNode(
                context=self._ctx.graph_attrib_context,
                triple=attr,
                token_aligns=aligns,
                sent=sent))
