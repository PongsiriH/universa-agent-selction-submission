import _import_root
import streamlit as st
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from universa.agents import BaseAgent
from universa.memory.chromadb.chromadb import ChromaDB
from solutions.custom_embedder import SentenceTransformerEF, TfIdfEF, EnsembleEF
from collections import Counter

# Set up full-screen layout
st.set_page_config(layout="wide")

# Composite score calculation function
def calculate_composite_score(agent, similarity_score):
    """
    Calculates the composite score for an agent based on various factors.
    """
    w0, w1, w2, w3 = 1, 0.5, -0.3 / 5, -0.2 / 5
    baseline_rating = 1
    k = 10
    adjusted_quality_score = (agent.average_rating * agent.rated_responses + baseline_rating * k) / (agent.rated_responses + k)
    log_popularity_score = math.log(agent.popularity + 1)
    total_estimated_cost = agent.input_cost + agent.output_cost
    response_time = agent.response_time
    composite_score = (
        w0 * adjusted_quality_score +
        w1 * log_popularity_score +
        w2 * total_estimated_cost +
        w3 * response_time
    )
    return composite_score, adjusted_quality_score, log_popularity_score, total_estimated_cost, response_time

# Cache agent loading and embedding setup
@st.cache_resource
def load_agents_and_embedding():
    """
    Loads agents from JSON files and fits embedding functions.
    """
    agents = []
    agents_dict = {}
    for agent_json in os.listdir('data/agents'):
        agent = BaseAgent.from_json(os.path.join("data/agents", agent_json))
        agent.model = None  # Ensure model is not loaded
        agents.append(agent)
        agents_dict[agent.name] = agent

    nn_ef = SentenceTransformerEF()
    tfidf_ef = TfIdfEF()
    tfidf_ef.fit([str(agent.description) for agent in agents])
    return agents, agents_dict, (nn_ef, tfidf_ef)

@st.cache_resource
def setup_chromadb(_agents, weights):
    """
    Sets up the ChromaDB with the given agents and weights.
    """
    chroma = ChromaDB(
        embedding_function=EnsembleEF(efs, weights),
        collection_name="example_collection",
        metadata={"hnsw:space": "cosine"}
    )
    chroma.add_data(
        documents=[str(agent.description) for agent in _agents],
        ids=[str(agent.name) for agent in _agents]
    )
    return chroma

def apply_softmax_sampling(df, exploration_rate=0.5, num_samples=100):
    """
    Applies softmax sampling to the agents based on their composite scores.
    """
    logits = df['Composite Score'].to_numpy() / exploration_rate
    # For numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    softmax_scores = exp_logits / np.sum(exp_logits)
    sampled_agents = np.random.choice(df["Agent name"], size=num_samples, p=softmax_scores)
    sample_counts = Counter(sampled_agents)
    df.loc[:, "Sample Count"] = df["Agent name"].map(sample_counts).fillna(0).astype(int)
    return df, softmax_scores

def process_query(query, chroma, agents_dict, n_results=5, exploration_rate=0.5, num_samples=100):
    """
    Processes the user's query and returns DataFrames for display.
    """
    result = chroma.query_data([query], n_results=n_results)
    
    # Return immediately if no results are found
    if not result["ids"][0]:
        return None, None, 0, False, None, None

    # Generate scores and create DataFrame
    scored_agents = score_agents(result, agents_dict)
    df = create_dataframe(scored_agents)

    # Check if a suitable agent was found
    max_similarity = df["Similarity Score"].max()
    found_suitable_agent = max_similarity >= 0.2

    # Apply softmax sampling and create display DataFrame
    df_display, softmax_scores = prepare_display_dataframe(df, exploration_rate, num_samples)

    # Filter and process agents with higher similarity
    filtered_agents_display, filtered_softmax_scores = process_filtered_agents(df, max_similarity, exploration_rate, num_samples)

    return df_display, filtered_agents_display, max_similarity, found_suitable_agent, softmax_scores, filtered_softmax_scores

def score_agents(result, agents_dict):
    """
    Scores each agent based on similarity and composite score.
    """
    scored_agents = []
    for agent_id, dissimilarity_score in zip(result["ids"][0], result["distances"][0]):
        agent = agents_dict[agent_id]
        similarity_score = 1 - dissimilarity_score
        composite_score, adjusted_quality_score, log_popularity_score, total_estimated_cost, response_time = calculate_composite_score(agent, None)
        adjusted_composite_score = similarity_score * composite_score
        scored_agents.append({
            "Agent name": agent.name,
            "Agent description": agent.description.replace('\n', ' '),
            "Composite Score": composite_score,
            "Adjusted Composite Score": adjusted_composite_score,
            "Similarity Score": similarity_score,
            "Quality Score": adjusted_quality_score,
            "Log Popularity": log_popularity_score,
            "Cost": total_estimated_cost,
            "Response Time": response_time
        })
    return sorted(scored_agents, key=lambda x: x["Composite Score"], reverse=True)

def create_dataframe(scored_agents):
    """
    Converts the scored agents list into a DataFrame.
    """
    columns = [
        "Agent name", "Agent description", "Composite Score", "Adjusted Composite Score", "Similarity Score",
        "Quality Score", "Log Popularity", "Cost", "Response Time"
    ]
    data = [
        [
            agent["Agent name"], agent["Agent description"], agent["Composite Score"], agent["Adjusted Composite Score"],
            agent["Similarity Score"], agent["Quality Score"], agent["Log Popularity"],
            agent["Cost"], agent["Response Time"]
        ] for agent in scored_agents
    ]
    return pd.DataFrame(data, columns=columns)

def prepare_display_dataframe(df, exploration_rate, num_samples):
    """
    Applies softmax sampling and prepares the display DataFrame.
    """
    df, softmax_scores = apply_softmax_sampling(df, exploration_rate, num_samples)
    
    # Round values for display
    df_display = df.copy()
    columns_to_round = [
        "Composite Score", "Adjusted Composite Score", "Similarity Score", "Quality Score",
        "Log Popularity", "Cost", "Response Time"
    ]
    df_display[columns_to_round] = df_display[columns_to_round].round(2)
    return df_display, softmax_scores

def process_filtered_agents(df, max_similarity, exploration_rate, num_samples):
    """
    Filters agents based on similarity and applies softmax sampling.
    """
    filtered_agents = df[df["Similarity Score"] > 0.8 * max_similarity]
    filtered_agents, filtered_softmax_scores = apply_softmax_sampling(filtered_agents, exploration_rate, num_samples)
    
    # Round values for display
    filtered_agents_display = filtered_agents.copy()
    columns_to_round = [
        "Composite Score", "Adjusted Composite Score", "Similarity Score", "Quality Score",
        "Log Popularity", "Cost", "Response Time"
    ]
    filtered_agents_display[columns_to_round] = filtered_agents_display[columns_to_round].round(2)
    return filtered_agents_display, filtered_softmax_scores

# Load agents and embeddings
agents, agents_dict, efs = load_agents_and_embedding()

# User Interface
st.title("Agent Selection Query Tool")

# Adjust layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Settings")
    semantic_weight = st.slider("Semantic Weight", 1e-5, 1.0, 0.5)
    lexical_weight = st.slider("Lexical Weight", 1e-5, 1.0, 0.5)
    exploration_rate = st.slider(
        "Exploration Rate", 0.01, 50.0, 0.5,
        help="Controls the randomness in sampling."
    )
    num_samples = st.number_input(
        "Number of Samples", min_value=1, max_value=1000, value=100,
        help="Number of agents to sample."
    )
    n_results = st.number_input(
        "Number of Results", min_value=1, max_value=100, value=5,
        help="Number of top agents to retrieve."
    )

    # Re-setup chromadb with new weights
    chroma = setup_chromadb(agents, [semantic_weight, lexical_weight])

with col2:
    col21, col22 = st.columns([3, 4])

    st.header("Query")
    query = st.text_input("Enter a query:")
    if st.button("Submit Query"):
        result = process_query(
            query, chroma, agents_dict, n_results, exploration_rate, num_samples
        )
        if result[0] is None:
            st.warning("No agents found matching your query.")
        else:
            (
                df_display, filtered_agents_display, max_similarity,
                found_suitable_agent, softmax_scores, filtered_softmax_scores
            ) = result

            st.write(f"Found suitable agent: **{found_suitable_agent}** with max similarity of **{round(max_similarity, 2)}**")


            with col21:
                st.dataframe(df_display)

                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.bar(df_display['Agent name'], softmax_scores)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Agents')
                ax.set_ylabel('Softmax Score')
                ax.set_title('Softmax Scores for Filtered Agents')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            with col22:
                st.dataframe(filtered_agents_display)

                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.bar(filtered_agents_display['Agent name'], filtered_softmax_scores)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Agents')
                ax.set_ylabel('Softmax Score')
                ax.set_title('Softmax Scores for Filtered Agents')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
