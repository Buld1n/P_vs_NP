import random
import numpy as np


# Greedy Algorithm for the Knapsack Problem
def knapsack_greedy(weights, values, capacity):
    """
    A greedy algorithm for the knapsack problem.
    Args:
    weights (list): List of weights of the items.
    values (list): List of values of the items.
    capacity (int): Maximum weight capacity of the knapsack.

    Returns:
    int: Total value of the knapsack.
    """
    value_per_weight = sorted([(v / w, w) for v, w in zip(values, weights)], reverse=True)
    total_value = 0
    for vpw, weight in value_per_weight:
        if capacity >= weight:
            capacity -= weight
            total_value += vpw * weight
    return total_value


# Basic Genetic Algorithm
def genetic_algorithm(population, fitness_function, mutation_rate, generations):
    """
    A basic genetic algorithm.
    Args:
    population (list): List of candidate solutions.
    fitness_function (function): Function to evaluate fitness of a solution.
    mutation_rate (float): Mutation rate.
    generations (int): Number of generations.

    Returns:
    list: Best solution found.
    """
    best_solution = None
    best_fitness = float('-inf')

    for _ in range(generations):
        # Evaluate population
        fitnesses = [fitness_function(individual) for individual in population]

        # Selection
        population = select(population, fitnesses)

        # Crossover
        population = crossover(population)

        # Mutation
        for individual in population:
            if random.random() < mutation_rate:
                mutate(individual)

        # Update best solution
        for individual, fitness in zip(population, fitnesses):
            if fitness > best_fitness:
                best_solution = individual
                best_fitness = fitness

    return best_solution


# Particle Swarm Optimization (PSO)
def particle_swarm_optimization(particles, fitness_function, iterations):
    """
    Particle Swarm Optimization (PSO) algorithm.
    Args:
    particles (list): List of particles (solutions).
    fitness_function (function): Function to evaluate fitness of a solution.
    iterations (int): Number of iterations.

    Returns:
    list: Best solution found.
    """
    best_global_solution = None
    best_global_fitness = float('-inf')

    for _ in range(iterations):
        # Evaluate particles
        for particle in particles:
            fitness = fitness_function(particle)
            # Update personal best
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            # Update global best
            if fitness > best_global_fitness:
                best_global_fitness = fitness
                best_global_solution = particle.position

        # Update velocities and positions
        for particle in particles:
            update_velocity(particle)
            update_position(particle)

    return best_global_solution


def select(population, fitnesses):
    """
    Selects individuals from the population based on their fitness.
    This simple implementation uses roulette wheel selection.
    """
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    selected_indices = np.random.choice(len(population), size=len(population), p=selection_probs)
    selected_population = [population[i] for i in selected_indices]
    return selected_population


def crossover(population):
    """
    Performs crossover on the population.
    This simple implementation uses one-point crossover.
    """
    next_generation = []
    for i in range(0, len(population), 2):
        parent1, parent2 = population[i], population[min(i+1, len(population)-1)]
        crossover_point = random.randint(1, len(parent1)-1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        next_generation.extend([child1, child2])
    return next_generation


def mutate(individual):
    """
    Performs mutation on an individual.
    This simple implementation flips a random bit.
    """
    mutation_point = random.randint(0, len(individual) - 1)
    individual[mutation_point] = 1 - individual[mutation_point]


def update_velocity(particle, global_best_position, w=0.5, c1=1, c2=1):
    """
    Updates the velocity of a particle.
    w is the inertia weight, c1 and c2 are cognitive and social coefficients.
    """
    r1, r2 = random.random(), random.random()
    cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
    social_velocity = c2 * r2 * (global_best_position - particle.position)
    particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity


def update_position(particle):
    """
    Updates the position of a particle based on its velocity.
    """
    particle.position += particle.velocity
