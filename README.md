# P vs NP Optimization Algorithms

This project contains a set of Python modules designed to explore and experiment with various optimization algorithms, particularly focusing on the challenges related to P vs NP problems. Each module is dedicated to a specific aspect of algorithm development and analysis.

## Modules

### 1. complexity_analysis.py

This module contains functions for analyzing the runtime complexity of algorithms.

- `evaluate_runtime(algorithm, data)`: Measures the execution time of a given algorithm with the provided input data.
- `compare_algorithms(algorithms, data)`: Compares the execution times of multiple algorithms on the same input data.

### 2. np_problems.py

This module generates instances of well-known NP problems.

- `generate_knapsack_data(n, max_weight, max_value)`: Generates data for the knapsack problem with specified parameters.
- `generate_sat_data(n_vars, n_clauses)`: Generates clauses for the SAT problem based on the number of variables and clauses.

### 3. optimization_search.py

Contains implementations of various heuristic search and optimization algorithms.

- `knapsack_greedy(weights, values, capacity)`: A greedy algorithm for the knapsack problem.
- `genetic_algorithm(population, fitness_function, mutation_rate, generations)`: Basic implementation of a genetic algorithm.
- `particle_swarm_optimization(particles, fitness_function, iterations)`: Basic implementation of the Particle Swarm Optimization algorithm.
- Helper functions for the genetic algorithm and PSO:
  - `select(population, fitnesses)`: Function for selecting the fittest individuals in the population.
  - `crossover(population)`: Function for performing crossover operations on the population.
  - `mutate(individual)`: Function for mutating individuals.
  - `update_velocity(particle, global_best_position, w, c1, c2)`: Updates the velocity of particles in PSO.
  - `update_position(particle)`: Updates the position of particles in PSO.


## Usage

Each module can be imported and used independently. Ensure you have the required Python environment and dependencies to run the modules. Example usage can be found within each module as comments or docstrings.

## Contributing

Contributions to improve the algorithms or to add new features are welcome. Please follow the standard pull request process for contributions.

## License

This project is open-sourced under the [MIT license](LICENSE).

