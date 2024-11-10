import _import_root
from solutions.selection_algorithm import Algorithm1
from benchmark.benchmark import Benchmark
from solutions.preprocess_agent_descriptions import process_benchmark_agents, repetitively_process_benchmark_agents

def algorithm1_with_raw_description():
    benchmark = Benchmark('benchmark.json')
    algorithm = Algorithm1(benchmark.agents, benchmark.agent_ids)
    algorithm.configure_embeddings_and_setup_chromadb('description')
    algorithm.ranked_by("Similarity Score")
    score = benchmark.validate(algorithm, verbose=False)
    return score

def algorithm1_with_processed_description():
    benchmark = Benchmark('benchmark_with_processed_description.json')
    algorithm = Algorithm1(benchmark.agents, benchmark.agent_ids)
    algorithm.configure_embeddings_and_setup_chromadb('processed_description', ngram_range=(2,5))
    algorithm.ranked_by("Similarity Score")
    score = benchmark.validate(algorithm, verbose=False)
    return score

if __name__ == "__main__":
    runs = 1
    score1_runs = []
    score2_runs = []

    for _ in range(runs):
        score1_runs.append(algorithm1_with_raw_description())
        
        process_benchmark_agents()
        score2_runs.append(algorithm1_with_processed_description())

    # Calculate the average for each score
    score1_avg = sum(score1_runs) / runs
    score2_avg = sum(score2_runs) / runs

    print(f'\nAverages:\n1 (avg): {score1_avg}\n2 (avg): {score2_avg}')

    for i in range(runs):
        print(f'Run {i+1} - ', end=' ')
        print(f'1: {score1_runs[i]: .4f}, ', end='')
        print(f'2: {score2_runs[i]: .4f}, ', end='')
        print()
