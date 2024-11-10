import json
import os.path as op

import _import_root  # noqa : E402

from tqdm import tqdm

from benchmark.selection import SelectionAlgorithm, ExampleAlgorithm


class Benchmark:
    def __init__(self, benchmark_filename) -> None:
        self.agents = open(op.join('benchmark', benchmark_filename), 'r').read()
        self.agents = json.loads(self.agents)
        self.agent_ids = [agent['object_id'] for agent in self.agents]

        self.queries = open(op.join('benchmark', 'queries.json'), 'r').read()
        self.queries = json.loads(self.queries)

        self.results = []

    def validate(self, algorithm: SelectionAlgorithm, verbose: bool = True) -> float:
        for query in tqdm(self.queries, desc="Validating queries"):
            result_id, result_agent = algorithm.select(query['query'])
            if query['object_id'] == result_id:
                self.results.append(True)
            else:
                self.results.append(False)
            if verbose:
                print(f"Query: {query['query']}")
                print(f"Result agent: {result_agent}")
                print(f"Expected agent: {query['agent']}")
                print(f"Score: {sum(self.results) / len(self.results)}")
                print()

        return sum(self.results) / len(self.results)

def example():
    benchmark = Benchmark('benchmark.json')
    print(benchmark.agents, benchmark.agent_ids)
    algorithm = ExampleAlgorithm(benchmark.agents, benchmark.agent_ids)
    score = benchmark.validate(algorithm, verbose=True)
    return score

if __name__ == '__main__':
    example()
