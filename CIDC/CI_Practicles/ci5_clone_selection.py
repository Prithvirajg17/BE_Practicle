import numpy as np

def objective_function(x):
    """Example objective function: Sphere function (minimization)."""
    return np.sum(x**2)

class CloneSelectionAlgorithm:
    def __init__(self, pop_size=10, clone_rate=2, mutation_rate=0.1, dimensions=2, generations=50):
        self.pop_size = pop_size
        self.clone_rate = clone_rate
        self.mutation_rate = mutation_rate
        self.dimensions = dimensions
        self.generations = generations
        self.population = np.random.uniform(-10, 10, (pop_size, dimensions))
    
    def evaluate_affinity(self):
        """Evaluate the fitness of each antibody (solution)."""
        return np.array([objective_function(x) for x in self.population])
    
    def select_best(self, affinities):
        """Select the top candidates based on affinity (lower is better)."""
        indices = np.argsort(affinities)[:self.pop_size // 2]  # Select top half
        return self.population[indices]
    
    def clone(self, best_solutions):
        """Clone the best solutions proportionally to their affinity."""
        num_clones = int(self.clone_rate * len(best_solutions))
        return np.repeat(best_solutions, num_clones, axis=0)
    
    def mutate(self, clones):
        """Apply mutation with intensity inversely proportional to affinity."""
        mutation_strength = self.mutation_rate * np.random.uniform(-1, 1, clones.shape)
        return clones + mutation_strength
    
    def run(self):
        """Run the clone selection process."""
        for gen in range(self.generations):
            affinities = self.evaluate_affinity()
            best_solutions = self.select_best(affinities)
            clones = self.clone(best_solutions)
            mutated_clones = self.mutate(clones)
            self.population = np.vstack((best_solutions, mutated_clones))  # Replace old population
            
            best_fitness = np.min(self.evaluate_affinity())
            print(f"Generation {gen + 1}: Best Fitness = {best_fitness:.5f}")
        
        return self.population[np.argmin(self.evaluate_affinity())]  # Return best solution

if __name__ == "__main__":
    csa = CloneSelectionAlgorithm()
    best_solution = csa.run()
    print("Best found solution:", best_solution)