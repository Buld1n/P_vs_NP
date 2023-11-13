import time


def evaluate_runtime(algorithm, data):
    start = time.time()
    algorithm(data)
    end = time.time()
    return end - start


def compare_algorithms(algorithms, data):
    times = {}
    for alg in algorithms:
        times[alg.__name__] = evaluate_runtime(alg, data)
    return times
