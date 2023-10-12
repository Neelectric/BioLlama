import json
import argparse
from collections import Counter

def parse_benchmark(benchmark):

    with open('benchmarks/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')

    data = json.loads(json_data)
    question_types = []
    for question in data["questions"]:
        pass

    print(len(data["questions"]))
    # count = Counter(data["questions"])
    # print(count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help="Name of the benchmark to parse.")
    args = parser.parse_args()
    parse_benchmark(args.benchmark)