import random
import numpy as np

class Chromosome:
    def __init__(self, genes):
        self.genes = genes  # each chromosome has 6 numbers

    # Convert the six-part gene into a float number
    def number_of_each_population(self):
        integer_part = self.genes[0] * 100 + self.genes[1] * 10 + self.genes[2]
        decimal_part = self.genes[3] * 0.1 + self.genes[4] * 0.01 + self.genes[5] * 0.001
        return integer_part + decimal_part

    def fitness(self, equation):
        x = self.number_of_each_population()
        return 1 / (1 + abs(equation(x)))

# genetic algorithm:
class GeneticAlgorithm:
    def __init__(self, population_size, equation):
        self.population_size = population_size
        self.equation = equation
        self.population = self._initialize_population()

    def _initialize_population(self):
        return [Chromosome([random.randint(0, 9) for _ in range(6)]) for _ in range(self.population_size)]

    def _select_parents(self):
        # Tournament selection
        parents = []
        for _ in range(self.population_size // 2):
            random_candidates = random.choices(self.population, k=2)
            parent = max(random_candidates, key=lambda c: c.fitness(self.equation))
            parents.append(parent)
        return parents

    def _crossover(self, parent1, parent2):
        # Single-point crossover
        crossover_point = random.randint(1, 5)
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
        return Chromosome(child1_genes), Chromosome(child2_genes)

    def _mutate(self, chromosome):
        # Randomly mutate a gene
        mutation_point = random.randint(0, 5)
        chromosome.genes[mutation_point] = random.randint(0, 9)
        return chromosome

    def _create_new_generation(self, parents):
        new_generation = []
        for _ in range(0, len(parents), 2):
            parent1, parent2 = parents[_], parents[_ + 1]
            child1, child2 = self._crossover(parent1, parent2)
            new_generation.extend([self._mutate(child1), self._mutate(child2)])
        return new_generation

    def run(self, generations=100):
        for _ in range(1000):
            parents = self._select_parents()
            self.population = self._create_new_generation(parents)

            # Sorted by fitness and print best solution
            self.population.sort(key=lambda c: c.fitness(self.equation), reverse=True)
            best_solution = self.population[0]
            print(f"Best solution: {best_solution.genes} -> {best_solution.number_of_each_population()} with fitness {best_solution.fitness(self.equation)}")

            if best_solution.fitness(self.equation) >= 0.95:  # termination
                break

        return self.population[0]

# Define your equation here
def equation(x):
    return 183 * (x**3) - 7.22 * (x**2) +15.5 *  x  -13.2 

# Run the algorithm
population_size = 100
genetic_algorithm = GeneticAlgorithm(population_size, equation)
solution = genetic_algorithm.run()
print("\n")
print(f"Found solution: {solution.number_of_each_population()} with genes {solution.genes}")
print(f"The result of the best chromosome is {equation(solution.number_of_each_population())}")
print("\n")
