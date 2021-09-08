#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:30:31 2021

@author: Rahul Chalamala
"""

# https://nerdimite.medium.com/introduction-to-genetic-algorithms-58263747b9a0

import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()


n = 30

lb = 0
ub = 100

population_size = 200
generations = 100
crossover_probability = 0.65
mutation_probability = 0.3


x = np.random.randint(lb, ub, n)
y = np.random.randint(lb, ub, n)
city_map = np.c_[x, y]


plt.figure()
plt.scatter(x, y)


cities = np.arange(n)


def initialize():
    population = []

    for _ in range(population_size):
        individual = cities.copy()
        rng.shuffle(individual)
        population.append(individual)

    return population


def fitness(solution):
    total = 0

    for i in range(n):
        total += np.linalg.norm(city_map[solution[i]] - city_map[solution[(i + 1) % n]])

    return total


def selection(population, weights):
    return rng.choice(population, p=weights)


def crossover(parent1, parent2):
    c1 = np.random.randint(0, n)
    c2 = np.random.randint(c1 + 1, n + 1)

    child = np.full(n, -1)

    child[c1:c2] = parent1[c1:c2]

    # BINARY SEARCH TO CHECK IF CHILD IN PARENT

    j = 0
    for i in range(c1):
        while parent2[j] in child:
            j += 1
        child[i] = parent2[j]

    for i in range(c2, len(parent2)):
        while parent2[j] in child:
            j += 1
        child[i] = parent2[j]

    return child


def mutation(individual):
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    mutant = individual.copy()

    mutant[i], mutant[j] = mutant[j], mutant[i]

    return mutant


def main():
    population = initialize()
    solution = [0, 0]
    original = 0

    for i in range(generations):
        print(i)

        population_fitness = [fitness(x) for x in population]

        best_fitness = min(population_fitness)
        best_index = population_fitness.index(best_fitness)
        best_individual = population[best_index]

        if i == 0 or best_fitness < solution[0]:
            solution = [best_fitness, best_individual]

        # print(population_fitness)
        if i == 0:
            original = min(population_fitness)

        if max(population_fitness) - min(population_fitness) < 1e-7:
            print("CONVERGED")
            break

        print(max(population_fitness))
        print(min(population_fitness))
        print(solution[0])

        new_population = []

        max_value = max(population_fitness)
        total_sum = max_value * population_size - sum(population_fitness)
        weights = [(max_value - x) / total_sum for x in population_fitness]

        for j in range(population_size):
            parent1 = selection(population, weights)
            parent2 = selection(population, weights)

            if np.random.rand() <= crossover_probability:
                child = crossover(parent1, parent2)
            else:
                child = rng.choice(np.array([parent1, parent2]))

            if np.random.rand() <= mutation_probability:
                child = mutation(parent1)

            new_population.append(child)

        population = new_population

    solution_points = np.array([city_map[x] for x in solution[1]] + [city_map[solution[1][0]]])
    x = solution_points[:, 0]
    y = solution_points[:, 1]

    plt.figure()
    plt.text(1, 1, fitness(solution[1]))
    plt.scatter(x, y)
    plt.plot(x, y)

    print(original)
    print(fitness(solution[1]))


if __name__ == "__main__":
    main()
