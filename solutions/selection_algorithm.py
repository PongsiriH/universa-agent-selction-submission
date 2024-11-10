import _import_root
import math
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any
from universa.memory.chromadb.chromadb import ChromaDB
from solutions.custom_embedder import (
    SentenceTransformerEF, TfIdfEF, EnsembleEF
)
from sentence_transformers import CrossEncoder
from benchmark.selection import SelectionAlgorithm
import uuid

class Algorithm1(SelectionAlgorithm):
    """
    Custom selection algorithm implementing composite scoring and softmax sampling.
    """
    def ranked_by(self, key="Similarity Score"):
        self._ranked_by = key
    
    def configure_embeddings_and_setup_chromadb(self, key='description', model_name="all-mpnet-base-v2", ngram_range=(2, 3), weights=[2, 1], distance_function="cosine", use_cross_encoder=False):
        """
        Configures embedding models and initializes the ChromaDB instance for agent similarity search.
        
        Args:
            key (str): The key in agent data used to create embeddings, typically 'description'.
            model_name (str): The name of the embedding model for sentence embeddings.
            ngram_range (tuple): The range of n-grams for TF-IDF embeddings.
            weights (list): The weights for combining different embedding models in the ensemble.
            distance_function (str): The distance metric for the embedding space in ChromaDB.
        """
        # Initialize the different embedding functions
        self.use_cross_encoder = use_cross_encoder
        self.nn_ef = SentenceTransformerEF(model_name)
        self.tfidf_ef = TfIdfEF(ngram_range)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Example model

        # Fit TF-IDF on agent descriptions
        descriptions = [str(agent[key]) for agent in self.agents]
        self.tfidf_ef.fit(descriptions)
        
        # Create ensemble embedding function with specified weights
        self.ensemble_ef = EnsembleEF([self.nn_ef, self.tfidf_ef], weights)
        
        # Initialize ChromaDB with the ensemble embedding function
        self.chroma = ChromaDB(
            embedding_function=self.ensemble_ef,
            collection_name=f"example_collection_{uuid.uuid4()}",
            metadata={"hnsw:space": distance_function}
        )
        self.chroma.collection.delete
        self.chroma.add_data(documents=descriptions, ids=self.ids)

    def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Initializes the selection algorithm with the given agents and their IDs.
        """
        self.agents = agents
        self.ids = ids
        self.agents_dict = {agent['name']: agent for agent in agents}
    
    def calculate_composite_score(self, agent: Dict[str, Any], similarity_score: float):
        """
        Calculates the composite score for an agent based on various factors.
        """
        w0, w1, w2, w3 = 1, 0.5, -0.3 / 5, -0.2 / 5
        baseline_rating = 1
        k = 10
        
        average_rating = agent.get('average_rating', None)
        rated_responses = agent.get('rated_responses', None)
        popularity = agent.get('popularity', None)
        input_cost = agent.get('input_cost', None)
        output_cost = agent.get('output_cost', None)
        response_time = agent.get('response_time', None)
        
        adjusted_quality_score = (
            (average_rating * rated_responses + baseline_rating * k) /
            (rated_responses + k)
        )
        log_popularity_score = math.log(popularity + 1)
        total_estimated_cost = input_cost + output_cost
        composite_score = similarity_score * (
            w0 * adjusted_quality_score +
            w1 * log_popularity_score +
            w2 * total_estimated_cost +
            w3 * response_time
        )
        return (
            composite_score, adjusted_quality_score, log_popularity_score,
            total_estimated_cost, response_time
        )
    
    def select(self, query: str, n_results = 5) -> Tuple[str, str]:
        """
        Selects the most relevant agent based on the provided query.

        Args:
            query (str): The input query used to search and rank agents.
            n_results (int, optional): The number of top agents to retrieve based on the initial query results. Defaults to 5.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing:
                - str: The unique identifier (Agent ID) of the best-matching agent.
                - str: The name of the best-matching agent.
                Returns (None, None) if no suitable agent is found.
        """                
        # Query for top n results
        result = self.chroma.query_data([query], n_results=n_results)
        
        # Calculate and store scores for each agent
        scored_agents = [
            self._score_agent(agent_id, dissimilarity_score)
            for agent_id, dissimilarity_score in zip(result["ids"][0], result["distances"][0])
        ]

        # Filter agents based on similarity threshold
        max_similarity = max(agent["Similarity Score"] for agent in scored_agents)
        filtered_agents = [agent for agent in scored_agents if agent["Similarity Score"] > 0.8 * max_similarity]

        # Apply cross-encoder scoring.
        if filtered_agents and self.use_cross_encoder:
            filtered_agents = self.cross_encoder_score(query, filtered_agents)
            filtered_agents.sort(key=lambda x: x["Cross-Encoder Score"], reverse=True)
        else:
            scored_agents = sorted(scored_agents, key=lambda x: x[self._ranked_by], reverse=True)

        # Return top agent or None if no agents found
        selected_agent = filtered_agents[0] if filtered_agents else None
        return (selected_agent["Agent ID"], selected_agent["Agent name"]) if selected_agent else (None, None)
    
    def _score_agent(self, agent_id: str, dissimilarity_score: float) -> dict:
        """
        Helper function to calculate and return agent scores as a dictionary.
        """
        agent_index = self.ids.index(agent_id)
        agent = self.agents[agent_index]
        similarity_score = 1 - dissimilarity_score
        composite_score, adjusted_quality_score, log_popularity_score, total_estimated_cost, response_time = self.calculate_composite_score(agent, similarity_score)
        
        return {
            "Agent ID": agent_id,
            "Agent name": agent['name'],
            "Agent description": agent['description'].replace('\n', ' '),
            "Composite Score": composite_score * similarity_score,
            "Similarity Score": similarity_score,
            "Quality Score": adjusted_quality_score,
            "Log Popularity": log_popularity_score,
            "Cost": total_estimated_cost,
            "Response Time": response_time
        }

    def cross_encoder_score(self, query: str, agents: List[Dict[str, Any]]):
        """
        Scores each agent in filtered_agents using a cross-encoder.
        """
        inputs = [(query, agent['Agent description']) for agent in agents]
        scores = self.cross_encoder.predict(inputs, show_progress_bar=False)
        for agent, score in zip(agents, scores):
            agent["Cross-Encoder Score"] = score
        return agents

    def apply_softmax_sampling(
        self, 
        agents: List[Dict[str, Any]],
        key = "Composite Score",
        exploration_rate: float = 0.5,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Applies softmax sampling to the agents based on their composite scores.

        Args:
            agents (List[Dict[str, Any]]): A list of agents, each represented by a dictionary key.
            exploration_rate (float, optional): The rate to control exploration; lower values increase exploration by flattening softmax probabilities. Defaults to 0.5.

        Returns:
            Tuple[List[Dict[str, Any]], np.ndarray]: 
                - List of agents with original data.
                - Numpy array of softmax scores corresponding to each agent.
        """
        # Normalize composite scores with exploration rate
        logits = np.array([agent[key] for agent in agents]) / exploration_rate
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        softmax_scores = exp_logits / np.sum(exp_logits)
        
        # Sample agent based on softmax probabilities
        sampled_agent_index = np.random.choice(len(agents), size=1, p=softmax_scores)[0]
        sampled_agent_uuid = agents[sampled_agent_index]["Agent ID"]

        return sampled_agent_index, sampled_agent_uuid