import _import_root
from solutions.selection_algorithm import Algorithm2
from benchmark.benchmark import Benchmark
from solutions.preprocess_agent_descriptions import process_benchmark_agents
from solutions.custom_embedder import (
    SentenceTransformerEF, TfIdfEF, SPLADEEmbeddingFunction
)
import pandas as pd
from tqdm import tqdm

def run_algorithm2(description_type, embedding_functions, verbose=False):
    benchmark = Benchmark('benchmark_with_processed_description.json')
    algorithm = Algorithm2(benchmark.agents, benchmark.agent_ids)
    algorithm.ranked_by(description_type, "Similarity Score")
    [ef.fit(algorithm.descriptions) for ef in embedding_functions if hasattr(ef, 'fit')]
    algorithm.configure_embeddings_and_setup_chromadb(embedding_functions)
    score = benchmark.validate(algorithm, verbose=verbose)
    return score

def main():
    input_keys = ['description', 'processed_description']
    embedding_functions_configs = {
        "mpnet": [None, SentenceTransformerEF("all-mpnet-base-v2")],
        "allMiniLM": [None, SentenceTransformerEF('sentence-transformers/all-MiniLM-L6-v2')],
        "splade": [None, SPLADEEmbeddingFunction()],
        "tfidf": [None, TfIdfEF(ngram_range=(2,3))],
        "tfidf_and_allMiniLM": [TfIdfEF(ngram_range=(2,3)), SentenceTransformerEF('sentence-transformers/all-MiniLM-L6-v2')],
        "splade_and_allMiniLM": [SPLADEEmbeddingFunction(), SentenceTransformerEF('sentence-transformers/all-MiniLM-L6-v2')],
        "tfidf_and_mpnet": [TfIdfEF(ngram_range=(2,3)), SentenceTransformerEF("all-mpnet-base-v2")],
        "splade_and_mpnet": [SPLADEEmbeddingFunction(), SentenceTransformerEF("all-mpnet-base-v2")],
        "tfidf_and_splade": [TfIdfEF(ngram_range=(2,3)), SPLADEEmbeddingFunction()],
        "allMiniLM_and_mpnet": [SentenceTransformerEF('sentence-transformers/all-MiniLM-L6-v2'), SentenceTransformerEF("all-mpnet-base-v2")],
    }
    scores = {f"{config_name}_{input_key}": [] 
              for config_name in embedding_functions_configs 
              for input_key in input_keys}

    for run in tqdm(range(runs), desc="Total Runs"):
        # UNCOMMENT the next line if you want new preprocessed descriptions. Note that it requires Ollama. 
        # process_benchmark_agents(run)

        for input_key in input_keys:
            input_key_run = f'{input_key}_{run}' if input_key.startswith('processed_description') else input_key

            for config_name, embedding_functions_config in tqdm(embedding_functions_configs.items(), leave=False, desc="Configurations"):
                score = run_algorithm2(input_key_run, embedding_functions_config, verbose=False)
                scores[f"{config_name}_{input_key}"].append(score)

    # Calculate the average score for each configuration-input key pair
    averages = {config_name: sum(score_list) / runs for config_name, score_list in scores.items()}
    # Convert scores to a DataFrame
    score_data = {}
    for config_input_key, score_list in scores.items():
        if len(score_list) == runs:
            score_data[config_input_key] = score_list + [sum(score_list) / runs]
    
    # Create a DataFrame with run columns and an average column
    df = pd.DataFrame.from_dict(score_data, orient='index', columns=[f"Run {i+1}" for i in range(runs)] + ["Average"])
    # Display the DataFrame
    print("\nScores Table:")
    print(df)
    df.to_csv('solutions/results.csv')

def analyze_best():
    input_key, best_config = "processed_description", [SPLADEEmbeddingFunction(), SentenceTransformerEF("all-mpnet-base-v2")]
    for run in range(runs):
        input_key_run = f'{input_key}_{run}' if input_key.startswith('processed_description') else input_key
        score = run_algorithm2(input_key_run, best_config, verbose=True)

if __name__ == "__main__":
    runs = 5
    # main()
    analyze_best()