import _import_root
import math
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any
from universa.memory.chromadb.chromadb import ChromaDB
from sentence_transformers import CrossEncoder
from benchmark.selection import SelectionAlgorithm
import uuid

def calculate_composite_score(agent: Dict[str, Any], similarity_score: float):
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

def softmax_sampling(
    agents: List[Dict[str, Any]],
    key = "Composite Score",
    exploration_rate: float = 1.0,
    n_samples :int=1
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Applies softmax sampling to the agents based on their composite scores.

    Args:
        agents (List[Dict[str, Any]]): A list of agents, each represented by a dictionary key.
        key (str): The key for composite scores.
        exploration_rate (float): Controls exploration; lower values increase exploration by flattening probabilities.
        n_samples (int): Number of samples to draw.

    Returns:
        Tuple[List[int], List[str]]: 
            - List of sampled indices.
            - List of sampled agent IDs.
    """
    # Normalize composite scores with exploration rate
    logits = np.array([agent[key] for agent in agents]) / exploration_rate
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    softmax_scores = exp_logits / np.sum(exp_logits)
    
    # Sample multiple agents based on softmax probabilities
    sampled_agent_indices = np.random.choice(len(agents), size=n_samples, p=softmax_scores)
    sampled_agent_ids = [agents[idx]["Agent ID"] for idx in sampled_agent_indices]

    return sampled_agent_indices, sampled_agent_ids, softmax_scores

class Algorithm2(SelectionAlgorithm):
    """
    Custom selection algorithm implementing composite scoring and softmax sampling.
    """
    def ranked_by(self, input_key='description', ranked_key="Similarity Score", ):
        self.rank_key = ranked_key
        self.input_key = input_key
        self.descriptions = [str(agent[self.input_key]) for agent in self.agents]
    
    def configure_embeddings_and_setup_chromadb(self, embedding_functions, distance_function="cosine"):
        """
        Configures embedding models and initializes the ChromaDB instance for agent similarity search.
        
        Args:
            key (str): The key in agent data used to create embeddings, typically 'description'.
            model_name (str): The name of the embedding model for sentence embeddings.
            ngram_range (tuple): The range of n-grams for TF-IDF embeddings.
            weights (list): The weights for combining different embedding models in the ensemble.
            distance_function (str): The distance metric for the embedding space in ChromaDB.
        """
        # Initialize embedding functions
        self.chromas: List[ChromaDB] = []
        for ef in embedding_functions:
            if ef is not None:     
                chroma = ChromaDB(
                    embedding_function=ef,
                    collection_name=f"collection_{uuid.uuid4()}",
                    metadata={"hnsw:space": distance_function}
                )
                chroma.add_data(documents=self.descriptions, ids=self.ids)
                self.chromas.append(chroma)
            else:
                self.chromas.append(None)
    
    def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Initializes the selection algorithm with the given agents and their IDs.
        """
        self.agents = agents
        self.ids = ids
        self.agents_dict = {agent['name']: agent for agent in agents}
        self.agents_id_dict = {agent['object_id']: agent for agent in agents}
        
    def select(self, query: str, n_results=5, k=60, return_best=True):
        """
        Selects the most relevant agent based on the provided query, using Reciprocal Rank Fusion (RRF).

        Args:
            query (str): The input query used to search and rank agents.
            n_results (int, optional): The number of top agents to retrieve based on the initial query results. Defaults to 5.
            k (int, optional): The constant used in the RRF score calculation. Defaults to 60.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing:
                - str: The unique identifier (Agent ID) of the best-matching agent.
                - str: The name of the best-matching agent.
                Returns (None, None) if no suitable agent is found.
        """
        rrf_scores = {}

        if self.chromas[0] is not None:
            lexical_results = self.chromas[0].query_data([query], n_results=n_results)

            # Process lexical results with RRF scoring
            for rank, (agent_id, dissimilarity_score) in enumerate(zip(lexical_results["ids"][0], lexical_results["distances"][0])):
                similarity_score = 1 - dissimilarity_score
                if similarity_score >= 0.1:  # Only include relevant lexical results
                    rrf_score = 1 / (rank + k) * math.sqrt(similarity_score) 
                    rrf_scores[agent_id] = rrf_scores.get(agent_id, 0) + rrf_score
                # print(f'lexical similarity_score={similarity_score}')

        # Process semantic results with RRF scoring
        semantic_results = self.chromas[1].query_data([query], n_results=n_results)
        for rank, (agent_id, dissimilarity_score) in enumerate(zip(semantic_results["ids"][0], semantic_results["distances"][0])):
            similarity_score = 1 - dissimilarity_score
            rrf_score = (1 / (rank + k)) * math.sqrt(similarity_score)
            rrf_scores[agent_id] = rrf_scores.get(agent_id, 0) + rrf_score
            # print(f'semantic similarity_score={similarity_score}')

        # Retrieve agent details and sort by combined RRF score
        scored_agents = [
            {"Agent ID": agent_id, "Combined RRF Score": score, "Agent name": self.agents_id_dict[agent_id]["name"]}
            for agent_id, score in rrf_scores.items()
        ]
        scored_agents.sort(key=lambda x: x["Combined RRF Score"], reverse=True)

        # Return top agent or None if no agents found
        if return_best:
            selected_agent = scored_agents[0] if scored_agents else None
            return (selected_agent["Agent ID"], selected_agent["Agent name"]) if selected_agent else (None, None)
        return scored_agents
    
    def score_agent(self, agent_id: str, dissimilarity_score: float) -> dict:
        """
        Helper function to calculate and return agent scores as a dictionary.
        """
        agent_index = self.ids.index(agent_id)
        agent = self.agents[agent_index]
        similarity_score = 1 - dissimilarity_score
        composite_score, adjusted_quality_score, log_popularity_score, total_estimated_cost, response_time = calculate_composite_score(agent, similarity_score)
        
        return {
            "Agent ID": agent_id,
            "Agent name": agent['name'],
            "Agent description": agent['description'].replace('\n', ' '),
            "Composite Score": composite_score,
            "Adjusted Composite Score": similarity_score * composite_score,
            "Similarity Score": similarity_score,
            "Quality Score": adjusted_quality_score,
            "Log Popularity": log_popularity_score,
            "Cost": total_estimated_cost,
            "Response Time": response_time
        }


# class Algorithm1(SelectionAlgorithm):
#     """
#     Custom selection algorithm implementing composite scoring and softmax sampling.
#     """
#     def ranked_by(self, input_key='description', ranked_key="Similarity Score", ):
#         self.rank_key = ranked_key
#         self.input_key = input_key
#         self.descriptions = [str(agent[self.input_key]) for agent in self.agents]
    
#     def configure_embeddings_and_setup_chromadb(self, embedding_function,
#                                                 distance_function="cosine", use_cross_encoder=False
#                                                 ):
#         """
#         Configures embedding models and initializes the ChromaDB instance for agent similarity search.
        
#         Args:
#             key (str): The key in agent data used to create embeddings, typically 'description'.
#             model_name (str): The name of the embedding model for sentence embeddings.
#             ngram_range (tuple): The range of n-grams for TF-IDF embeddings.
#             weights (list): The weights for combining different embedding models in the ensemble.
#             distance_function (str): The distance metric for the embedding space in ChromaDB.
#         """
#         # Initialize the different embedding functions
#         self.use_cross_encoder = use_cross_encoder
#         if self.use_cross_encoder:
#             self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        
#         # Initialize ChromaDB with the ensemble embedding function
#         self.chroma = ChromaDB(
#             embedding_function=embedding_function,
#             collection_name=f"example_collection_{uuid.uuid4()}",
#             metadata={"hnsw:space": distance_function}
#         )
#         self.chroma.collection.delete
#         self.chroma.add_data(documents=self.descriptions, ids=self.ids)

#     def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
#         """
#         Initializes the selection algorithm with the given agents and their IDs.
#         """
#         self.agents = agents
#         self.ids = ids
#         self.agents_dict = {agent['name']: agent for agent in agents}
        
    
#     def calculate_composite_score(self, agent: Dict[str, Any], similarity_score: float):
#         """
#         Calculates the composite score for an agent based on various factors.
#         """
#         w0, w1, w2, w3 = 1, 0.5, -0.3 / 5, -0.2 / 5
#         baseline_rating = 1
#         k = 10
        
#         average_rating = agent.get('average_rating', None)
#         rated_responses = agent.get('rated_responses', None)
#         popularity = agent.get('popularity', None)
#         input_cost = agent.get('input_cost', None)
#         output_cost = agent.get('output_cost', None)
#         response_time = agent.get('response_time', None)
        
#         adjusted_quality_score = (
#             (average_rating * rated_responses + baseline_rating * k) /
#             (rated_responses + k)
#         )
#         log_popularity_score = math.log(popularity + 1)
#         total_estimated_cost = input_cost + output_cost
#         composite_score = similarity_score * (
#             w0 * adjusted_quality_score +
#             w1 * log_popularity_score +
#             w2 * total_estimated_cost +
#             w3 * response_time
#         )
#         return (
#             composite_score, adjusted_quality_score, log_popularity_score,
#             total_estimated_cost, response_time
#         )
    
#     def select(self, query: str, n_results = 5) -> Tuple[str, str]:
#         """
#         Selects the most relevant agent based on the provided query.

#         Args:
#             query (str): The input query used to search and rank agents.
#             n_results (int, optional): The number of top agents to retrieve based on the initial query results. Defaults to 5.

#         Returns:
#             Tuple[Optional[str], Optional[str]]: A tuple containing:
#                 - str: The unique identifier (Agent ID) of the best-matching agent.
#                 - str: The name of the best-matching agent.
#                 Returns (None, None) if no suitable agent is found.
#         """                
#         # Query for top n results
#         result = self.chroma.query_data([query], n_results=n_results)
        
#         # Calculate and store scores for each agent
#         scored_agents = [
#             self._score_agent(agent_id, dissimilarity_score)
#             for agent_id, dissimilarity_score in zip(result["ids"][0], result["distances"][0])
#         ]

#         # Filter agents based on similarity threshold
#         max_similarity = max(agent["Similarity Score"] for agent in scored_agents)
#         filtered_agents = [agent for agent in scored_agents if agent["Similarity Score"] > 0.8 * max_similarity]
#         self.filtered_agents = filtered_agents # For printing.

#         # Apply cross-encoder scoring.
#         if filtered_agents and self.use_cross_encoder:
#             filtered_agents = self.cross_encoder_score(query, filtered_agents)
#             filtered_agents.sort(key=lambda x: x["Cross-Encoder Score"], reverse=True)
#         else:
#             scored_agents = sorted(scored_agents, key=lambda x: x[self.rank_key], reverse=True)

#         # Return top agent or None if no agents found
#         selected_agent = filtered_agents[0] if filtered_agents else None
#         return (selected_agent["Agent ID"], selected_agent["Agent name"]) if selected_agent else (None, None)
    
#     def _score_agent(self, agent_id: str, dissimilarity_score: float) -> dict:
#         """
#         Helper function to calculate and return agent scores as a dictionary.
#         """
#         agent_index = self.ids.index(agent_id)
#         agent = self.agents[agent_index]
#         similarity_score = 1 - dissimilarity_score
#         composite_score, adjusted_quality_score, log_popularity_score, total_estimated_cost, response_time = self.calculate_composite_score(agent, similarity_score)
        
#         return {
#             "Agent ID": agent_id,
#             "Agent name": agent['name'],
#             "Agent description": agent['description'].replace('\n', ' '),
#             "Composite Score": composite_score * similarity_score,
#             "Similarity Score": similarity_score,
#             "Quality Score": adjusted_quality_score,
#             "Log Popularity": log_popularity_score,
#             "Cost": total_estimated_cost,
#             "Response Time": response_time
#         }

#     def cross_encoder_score(self, query: str, agents: List[Dict[str, Any]]):
#         """
#         Scores each agent in filtered_agents using a cross-encoder.
#         """
#         inputs = [(query, agent['Agent description']) for agent in agents]
#         scores = self.cross_encoder.predict(inputs, show_progress_bar=False)
#         for agent, score in zip(agents, scores):
#             agent["Cross-Encoder Score"] = score
#         return agents
