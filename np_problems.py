import random


def generate_knapsack_data(n, max_weight, max_value):
    weights = [random.randint(1, max_weight) for _ in range(n)]
    values = [random.randint(1, max_value) for _ in range(n)]
    capacity = sum(weights) // 2
    return weights, values, capacity


def generate_sat_data(n_vars, n_clauses):
    clauses = []
    for _ in range(n_clauses):
        n_vars_in_clause = random.randint(1, n_vars)
        clause = [(random.randint(1, n_vars), random.choice([True, False])) for _ in range(n_vars_in_clause)]
        clauses.append(clause)
    return clauses
