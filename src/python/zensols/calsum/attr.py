"""Clinical graph nodes.

"""
__author__ = 'Paul Landes'

from typing import ClassVar
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from zensols.calamr.attr import ConceptGraphNode
from . import Cui

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False)
class ClinicalConceptGraphNode(ConceptGraphNode):
    """Contains clinical concept (CUI) information from entities that were
    linked to aligned tokens.

    """
    ATTRIB_TYPE: ClassVar[str] = 'clinical concept'
    """The attribute type this class represents."""

    cui: Cui = field()
    """A clinical concept ID and its description."""

    def __post_init__(self):
        super().__post_init__()
        self.embedding_resource.populate_cui_description(self.cui)

    def _get_embedding(self) -> Tensor:
        sup: Tensor = super()._get_embedding()
        cui: Tensor = self.embedding_resource.get_cui_embedding(self.cui)
        return torch.stack((sup, cui), dim=0).mean(dim=0)

    def _get_description(self) -> str:
        par: str = super()._get_description()
        s: str = f'{par} ({self.cui})'
        if self.cui.entity is not None:
            s = f'{s}: {self.cui.entity.definition}'
        return s

    def _get_label(self) -> str:
        return super()._get_description()
