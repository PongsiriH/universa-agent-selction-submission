from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingFunction(ABC):
    """
    Abstract base class for embedding functions.
    """

    @abstractmethod
    def create_embeddings(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        """
        Create embeddings for the given input texts.

        Args:
            texts (List[str]): A list of input text strings.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        pass