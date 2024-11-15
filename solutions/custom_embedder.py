import _import_root
from universa.utils.logs import get_logger
from universa.memory.embedding_functions.base_embedder import BaseEmbeddingFunction

from typing import List

import re
import numpy as np
import torch
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


class SentenceTransformerEF(BaseEmbeddingFunction):
    """
    Embedding function using SentenceTransformer.

    Args:
        model_name (str): The name of the SentenceTransformer model to use.
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.ef = SentenceTransformer(model_name, device='cpu')
        self.device = 'cpu'
        self.ef.to(self.device)
        self.logger = get_logger(self.__class__.__name__)

    def create_embeddings(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """
        Create embeddings for a list of texts using SentenceTransformer.

        Args:
            texts (List[str]): A list of input text strings.
            batch_size (int): The number of texts to process in each batch.

        Returns:
            List[List[float]]: A list of embedding vectors.

        Raises:
            Exception: If an error occurs during embedding creation.
        """
        try:
            embeddings = self.ef.encode(
                texts,
                show_progress_bar=False,
                batch_size=batch_size,
                device=self.device
            )
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error in creating embeddings: {e}")
            raise e


class TfIdfEF(BaseEmbeddingFunction):
    """
    TF-IDF-based embedding function. Uses TF-IDF scores as embeddings.

    Args:
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams.
    """

    def __init__(self, ngram_range: tuple = (2, 5)):
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        self.stemmer = SnowballStemmer('english') # Assume English language.
        self.is_fitted = False
        self.logger = get_logger(self.__class__.__name__)

    def fit(self, corpus: List[str]) -> None:
        """
        Fit the TF-IDF model using the given corpus.

        Args:
            corpus (List[str]): A list of text documents to fit the TF-IDF model.

        Raises:
            Exception: If an error occurs during fitting.
        """
        preprocessed_corpus = self._preprocess_texts(corpus)
        try:
            self.vectorizer.fit(preprocessed_corpus)
            self.is_fitted = True
        except Exception as e:
            self.logger.error(f"Error in fitting the TF-IDF model: {e}")
            raise e

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts. Requires that `fit` has been called.

        Args:
            texts (List[str]): A list of input text strings.

        Returns:
            List[List[float]]: A list of embedding vectors.

        Raises:
            ValueError: If the TF-IDF model has not been fitted.
            Exception: If an error occurs during embedding creation.
        """
        if not self.is_fitted:
            raise ValueError("The TF-IDF model is not fitted. Call `fit` with a corpus before creating embeddings.")

        preprocessed_texts = self._preprocess_texts(texts)
        try:
            embeddings = self.vectorizer.transform(preprocessed_texts).toarray()
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error in creating embeddings: {e}")
            raise e

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts by lowercasing, removing special characters, and applying stemming.

        Args:
            texts (List[str]): A list of input text strings.

        Returns:
            List[str]: A list of preprocessed text strings.
        """
        processed_texts = []
        for text in texts:
            # Lowercase and remove non-alphanumeric characters
            cleaned_text = re.sub(r'[^a-z0-9\s]', '', text.lower())
            # Apply stemming to each word
            stemmed_words = [self.stemmer.stem(word) for word in cleaned_text.split()]
            processed_text = ' '.join(stemmed_words)
            processed_texts.append(processed_text)
        return processed_texts
    
class SPLADEEmbeddingFunction(BaseEmbeddingFunction):
    """
    SPLADE embedding function subclass that creates sparse embeddings using
    a pre-trained SPLADE model.
    """

    def __init__(self, model_id='naver/splade-cocondenser-ensembledistil'):
        """
        Initializes the SPLADE embedding function with a specified model.

        Args:
            model_id (str): The Hugging Face model ID for SPLADE.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)

    def create_embeddings(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        """
        Create sparse embeddings for the given input texts.

        Args:
            texts (List[str]): A list of input text strings.

        Returns:
            List[List[float]]: A list of sparse embedding vectors.
        """
        embeddings = []

        for text in texts:
            # Tokenize input text
            tokens = self.tokenizer(text, return_tensors='pt')
            
            # Process tokens through the model to get logits
            output = self.model(**tokens)
            
            # Calculate sparse vector for SPLADE
            sparse_vec = torch.max(
                torch.log1p(torch.relu(output.logits)) * tokens['attention_mask'].unsqueeze(-1),
                dim=1
            )[0].squeeze()
            
            embeddings.append(sparse_vec.tolist())  # Convert tensor to list for compatibility

        return embeddings
