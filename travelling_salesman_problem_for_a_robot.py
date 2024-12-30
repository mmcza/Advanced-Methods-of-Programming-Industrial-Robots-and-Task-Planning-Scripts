############################################################################################################
#   Mostly done by @dariak153, I've fixed some issues and added some finishing touches to the code         #
############################################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.point_to_index = {}

    def create_individual(self, points):
        return ['p0'] + random.sample(points, len(points)) + ['p0']

    def create_population(self, points):
        self.population = [self.create_individual(points) for _ in range(self.population_size)]
        self.point_to_index = {point: i for i, point in enumerate(['p0'] + points)}  # Index home as 0

    def calculate_total_time(self, individual, time_matrix):
        total_time = 0
        for i in range(len(individual) - 1):
            index_a = self.point_to_index[individual[i]]
            index_b = self.point_to_index[individual[i + 1]]
            total_time += time_matrix[index_a][index_b]
        return total_time

    def select_parents(self, time_matrix):
        scores = [(self.calculate_total_time(individual, time_matrix), individual) for individual in self.population]
        scores.sort(key=lambda x: x[0])
        return [individual for _, individual in scores[:2]]

    def crossover(self, parent1, parent2):
        split = random.randint(1, len(parent1) - 2)
        child_middle = parent1[1:split]
        remaining_genes = [gene for gene in parent2 if gene not in child_middle and gene != 'p0']
        child = ['p0'] + child_middle + remaining_genes + ['p0']
        return child

    def mutate(self, individual):
        for idx in range(1, len(individual) - 1):
            if random.random() < self.mutation_rate:
                swap_idx = random.randint(1, len(individual) - 2)
                individual[idx], individual[swap_idx] = individual[swap_idx], individual[idx]

    def evolve(self, time_matrix):
        for generation in range(self.generations):
            parents = self.select_parents(time_matrix)
            next_generation = parents.copy()
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)
            self.population = next_generation

    def best_individual(self, time_matrix):
        best = min(self.population, key=lambda ind: self.calculate_total_time(ind, time_matrix))
        return best, self.calculate_total_time(best, time_matrix)

def calculate_time_matrix(points, connections, obstacles, home, constant=0.5):
    num_points = len(points) + 1  # Include home as p0
    time_matrix = np.full((num_points, num_points), 0.001)

    # Handle home point
    time_matrix[0][0] = 0

    # Calculate distances from home
    for i, point in enumerate(points, start=1):
        coord_a = np.array(home)
        coord_b = np.array(data["points"][point])
        dist = np.linalg.norm(coord_a - coord_b)
        time_matrix[0][i] = dist * constant  # Time from home to point
        time_matrix[i][0] = dist * constant  # Time from point to home

    # Calculate distances between points
    for i, point_a in enumerate(points, start=1):
        for j, point_b in enumerate(points, start=1):
            if i != j:
                coord_a = np.array(data["points"][point_a])
                coord_b = np.array(data["points"][point_b])
                dist = np.linalg.norm(coord_a - coord_b)
                time_matrix[i][j] = dist * constant
                time_matrix[j][i] = dist * constant
                # Check if there is an obstacle between the points
                for obstacle in obstacles:
                    min_x = obstacle[0][0]
                    max_x = obstacle[1][0]
                    min_y = obstacle[0][1]
                    max_y = obstacle[2][1]

                    points_to_check = [coord_a + (coord_b - coord_a) * t for t in np.linspace(0, 1, 100)]
                    for point in points_to_check:
                        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                            time_matrix[i][j] = 100000 * constant
                            time_matrix[j][i] = 100000 * constant
                            print(point)
                            break

    for conn in connections:
        index_a = points.index(conn[0]) + 1
        index_b = points.index(conn[1]) + 1
        time_matrix[index_a][index_b] = 0.001  # Very short time for direct connections
        time_matrix[index_b][index_a] = 0.001

    return time_matrix

def visualize(points, connections, obstacles, home, best_route=None):
    plt.figure(figsize=(10, 8))

    # Draw home point
    plt.scatter(home[0], home[1], color='orange', label='Home (p0)', s=100)

    # Draw points
    for point in points:
        plt.scatter(data["points"][point][0], data["points"][point][1], label=point)

    # Draw connections
    for conn in connections:
        start = np.array(data["points"][conn[0]])
        end = np.array(data["points"][conn[1]])
        plt.plot([start[0], end[0]], [start[1], end[1]], 'g--', label='Connection' if 'Connection' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Draw obstacles
    for obstacle in obstacles:
        poly = plt.Polygon(obstacle, color='r', alpha=0.5, label='Obstacle' if 'Obstacle' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.gca().add_patch(poly)

    # Draw best route
    if best_route:
        for i in range(len(best_route) - 1):
            if best_route[i] == 'p0':
                start = np.array(home)
            else:
                start = np.array(data["points"][best_route[i]])
            if best_route[i + 1] == 'p0':
                end = np.array(home)
            else:
                end = np.array(data["points"][best_route[i + 1]])
            plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2, label='Best Route' if i == 0 else "")
            plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], head_width=0.5, head_length=0.5, fc='blue', ec='blue')

    plt.xlim(-5, 25)
    plt.ylim(-5, 30)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Genetic Algorithm - Route Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

# Data from board
data = {
    "home": [0, 0],
    "points": {
        "p1": [3, 3],
        "p2": [3, 10],
        "p3": [11, 3],
        "p4": [16, 7],
        "p5": [8, 12],
        "p6": [16, 12],
        "p7": [16, 15],
        "p8": [18, 19],
        "p9": [15, 27],
        "p10": [3, 24],
        "p11": [9, 27],
        "p12": [5, 20],
        "p13": [10, 21]
    },
    "connections": [
        ["p1", "p2"],
        ["p3", "p4"],
        ["p5", "p6"],
        ["p6", "p7"],
        ["p8", "p9"],
        ["p10", "p11"],
        ["p12", "p13"]
    ],
    "obstacles": [
        [[11, 9], [17, 9], [17, 10], [11, 10]]
    ]
}

data["points"]["p0"] = data["home"]

def main():

    points = list(data["points"].keys())
    points.remove('p0')
    home = data["home"]
    time_matrix = calculate_time_matrix(points, data["connections"], data["obstacles"], home)

    population_size = 400
    generations = 2000
    mutation_rate = 0.01

    ga = GeneticAlgorithm(population_size, generations, mutation_rate)
    ga.create_population(points)
    ga.evolve(time_matrix)

    best_route, best_time = ga.best_individual(time_matrix)
    print("Best route:", ' -> '.join(best_route))
    print("Total time:", round(best_time, 2))
    visualize(points, data["connections"], data["obstacles"], home, best_route)

if __name__ == '__main__':
    main()