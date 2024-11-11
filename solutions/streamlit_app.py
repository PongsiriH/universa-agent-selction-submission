import os
import _import_root
import pysqlite3
import streamlit as st
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from selection_algorithm import Algorithm2, softmax_sampling
from custom_embedder import SentenceTransformerEF, SPLADEEmbeddingFunction
import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
@st.cache_resource
def setup_selection_algorithm():
    def load_agents():
        agents = []
        agent_ids = []
        for agent_json in os.listdir('data/agents'):
            with open(os.path.join("data/agents", agent_json), 'r') as f:
                agent = json.load(f)
            agents.append(agent)
            agent['object_id'] = agent['uuid'] # for compatibility
            agent_ids.append(agent['uuid'])
        return agents, agent_ids

    embedding_functions = [SPLADEEmbeddingFunction(), SentenceTransformerEF("all-mpnet-base-v2")]
    agents, agent_ids = load_agents()
    selection = Algorithm2(agents, agent_ids)
    selection.ranked_by('description')
    selection.configure_embeddings_and_setup_chromadb(embedding_functions=embedding_functions)
    return selection

selection = setup_selection_algorithm()

# Query input section
st.write("### Enter a New Query")
query = st.text_input("Enter a query:", "")
if st.button("Submit Query"):
    
    # 1. Retrieve agent results based on the query
    results = selection.select(query, return_best=False)
    agent_info_list = [
        selection.score_agent(result['Agent ID'], 1 - result['Combined RRF Score'])
        for result in results
    ]

    col1, col2 = st.columns([1, 1])
    # 2. Top N Agents - Display agent information table with sample counts
    with col1:
        st.write("### Top N Agents (with Sample Counts)")
        agent_info_df = pd.DataFrame(agent_info_list)

        # Perform softmax sampling on "Adjusted Composite Score" and add 'Sample Count' column
        sampled_agent_index, sampled_agent_ids, softmax_scores = softmax_sampling(
            agent_info_list, "Adjusted Composite Score", exploration_rate=1.0, n_samples=100
        )
        sample_counts = Counter(sampled_agent_ids)
        agent_info_df['Sample Count'] = agent_info_df['Agent ID'].map(sample_counts).fillna(0).astype(int)

        # Plot softmax scores for Top N Agents
        fig, ax = plt.subplots(figsize=(10, 4))
        agent_names = [agent["Agent name"] for agent in agent_info_list]
        ax.bar(agent_names, softmax_scores)
        ax.set_xlabel("Agents")
        ax.set_ylabel("Softmax Score")
        ax.set_title("Softmax Scores for Top N Agents")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        column_order = ["Agent ID", "Agent name", "Sample Count", "Composite Score", "Adjusted Composite Score", 
                        "Similarity Score", "Quality Score", "Log Popularity", "Cost", "Response Time", "Agent description"]
        agent_info_df = agent_info_df[column_order]
        # Display the updated DataFrame without 'Agent ID' column
        st.dataframe(agent_info_df.drop(columns=["Agent ID"]))

    # 3. Filtered Agents - Display agents filtered by similarity score with updated sample counts
    with col2:
        st.write("### Filtered Agents (Similarity Score > 80% of Max)")
        # Filter agents with similarity score > 80% of max similarity score
        filtered_agent_info_df = agent_info_df[agent_info_df['Similarity Score'] > 0.8 * agent_info_df['Similarity Score'].max()]

        # Perform softmax sampling on "Composite Score" for filtered agents
        filtered_agent_list = filtered_agent_info_df.to_dict('records')
        _, filtered_sampled_agent_ids, filtered_softmax_scores = softmax_sampling(
            filtered_agent_list, "Composite Score", exploration_rate=1.0, n_samples=100
        )
        filtered_sample_counts = Counter(filtered_sampled_agent_ids)
        filtered_agent_info_df['Sample Count'] = filtered_agent_info_df['Agent ID'].map(filtered_sample_counts).fillna(0).astype(int)

        # Plot softmax scores for Filtered Agents
        fig, ax = plt.subplots(figsize=(10, 4))
        filtered_agent_names = filtered_agent_info_df['Agent name'].tolist()
        ax.bar(filtered_agent_names, filtered_softmax_scores)
        ax.set_xlabel("Filtered Agents")
        ax.set_ylabel("Softmax Score")
        ax.set_title("Softmax Scores for Filtered Agents")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        # Display the filtered DataFrame without 'Agent ID' column
        column_order = ["Agent ID", "Agent name", "Sample Count", "Composite Score", "Adjusted Composite Score", 
                "Similarity Score", "Quality Score", "Log Popularity", "Cost", "Response Time", "Agent description"]
        filtered_agent_info_df = agent_info_df[column_order]
        st.dataframe(filtered_agent_info_df.drop(columns=["Agent ID"]))