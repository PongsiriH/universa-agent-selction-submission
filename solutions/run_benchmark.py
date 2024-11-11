import _import_root
from solutions.selection_algorithm import Algorithm1, Algorithm2
from benchmark.benchmark import Benchmark
from solutions.preprocess_agent_descriptions import process_benchmark_agents, repetitively_process_benchmark_agents
from solutions.custom_embedder import (
    SentenceTransformerEF, TfIdfEF, EnsembleEF, SPLADEEmbeddingFunction
)

def algorithm1_with_raw_description():
    benchmark = Benchmark('benchmark.json')
    algorithm = Algorithm1(benchmark.agents, benchmark.agent_ids)
    algorithm.ranked_by('description', "Similarity Score")
    tfidf = TfIdfEF()
    tfidf.fit(algorithm.descriptions)
    algorithm.configure_embeddings_and_setup_chromadb(EnsembleEF([SentenceTransformerEF("all-mpnet-base-v2"), tfidf], weights=[1, 0.5]))
    # algorithm.configure_embeddings_and_setup_chromadb(SentenceTransformerEF("all-mpnet-base-v2"))
    score = benchmark.validate(algorithm, verbose=False)
    return score

def algorithm1_with_processed_description():
    benchmark = Benchmark('benchmark_with_processed_description.json')
    algorithm = Algorithm1(benchmark.agents, benchmark.agent_ids)
    algorithm.ranked_by('processed_description', "Similarity Score")
    tfidf = TfIdfEF((2,3))
    tfidf.fit(algorithm.descriptions)
    algorithm.configure_embeddings_and_setup_chromadb(EnsembleEF([SentenceTransformerEF("all-mpnet-base-v2"), tfidf], [2, 1]))
    # algorithm.configure_embeddings_and_setup_chromadb(SentenceTransformerEF("all-mpnet-base-v2"))
    score = benchmark.validate(algorithm, verbose=True)
    return score

def algorithm2_with_processed_description():
    benchmark = Benchmark('benchmark_with_processed_description.json')
    algorithm = Algorithm2(benchmark.agents, benchmark.agent_ids)
    algorithm.ranked_by('processed_description', "Similarity Score")
    algorithm.configure_embeddings_and_setup_chromadb([SPLADEEmbeddingFunction(), SentenceTransformerEF("all-mpnet-base-v2")], )
    score = benchmark.validate(algorithm, verbose=True)
    return score

if __name__ == "__main__":
    runs = 5
    score1_runs = []
    score2_runs = []
    score3_runs = []

    for _ in range(runs):
        score1_runs.append(algorithm1_with_raw_description())
        
        process_benchmark_agents() # require ollma with mistral model.
        score2_runs.append(algorithm1_with_processed_description())
        score3_runs.append(algorithm2_with_processed_description())

    # Calculate the average for each score
    score1_avg = sum(score1_runs) / runs
    score2_avg = sum(score2_runs) / runs
    score3_avg = sum(score3_runs) / runs

    print(f'\nAverages:\n1 (avg): {score1_avg}\n2 (avg): {score2_avg}\n3 (avg): {score3_avg}')

    for i in range(runs):
        print(f'Run {i+1} - ', end=' ')
        print(f'1: {score1_runs[i]: .4f}, ', end='')
        print(f'2: {score2_runs[i]: .4f}, ', end='')
        print(f'3: {score3_runs[i]: .4f}, ', end='')
        print()
