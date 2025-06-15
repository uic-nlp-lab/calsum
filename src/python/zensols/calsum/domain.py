"""Application domain classes.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from torch import Tensor
from zensols.util import APIError
from zensols.deepnlp.transformer import WordPieceFeatureDocument
from zensols.mednlp.lib import MedicalLibrary
from zensols.mednlp.entlink import Entity
from zensols.calamr import EmbeddingResource

logger = logging.getLogger(__name__)


class CalsumError(APIError):
    """Thrown for any application level error.

    """


@dataclass
class Cui(object):
    """A clinical concept ID and its description.

    """
    id: str = field()
    """The CUI ID, which has the form ``C<ID>``."""

    preferred_name: str = field()
    """A short description, which is taken from the ``preferred_name``
    :class:`~zensols.nlp.tok.FeatureToken` featureID).

    """
    entity: Entity = field(default=None)
    """"""

    def __str__(self):
        return f'{self.id}: {self.preferred_name}'


@dataclass
class ClinicalEmbeddingResource(EmbeddingResource):
    medical_library: MedicalLibrary = field(default=None)
    """If set, populate :obj:`.Cui.entity` and use definitions for sentence."""

    def populate_cui_description(self, cui: Cui):
        """Add :obj:`.Cui.entity` if :obj:`medical_library` is set."""
        if self.medical_library is not None:
            cui.entity = self.medical_library.get_linked_entity(cui.id)

    def get_cui_embedding(self, cui: Cui) -> Tensor:
        """Return the mean of the token embeddings of ``text``."""
        text: str = cui.preferred_name
        if cui.entity is not None and cui.entity.definition is not None:
            text = cui.entity.definition
        wp_doc: WordPieceFeatureDocument = self.get_word_piece_document(text)
        return wp_doc.embedding[0]
