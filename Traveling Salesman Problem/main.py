#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:30:31 2021

@author: Rahul Chalamala
"""

# https://nerdimite.medium.com/introduction-to-genetic-algorithms-58263747b9a0

import numpy as np
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng()


n = 30

lb = 0
ub = 1000

population_size = 100
generations = 1000
crossover_probability = 0.65
mutation_probability = 0.4
elitism = 0.8


x = np.random.randint(lb, ub, n)
y = np.random.randint(lb, ub, n)
city_map = np.c_[x, y]


dt = np.dtype('u1')
fdt = np.dtype('f8')

cities = np.arange(n, dtype=dt)


def initialize():
    population = np.full(shape=(population_size, n), fill_value=cities, dtype=object)
    
    return rng.permuted(population, axis=1)


def fitness(solution):
    return np.sum([np.linalg.norm(city_map[solution[i]] - city_map[solution[(i + 1) % n]]) for i in range(n)], dtype=fdt)


def selection(population, weights):
    return rng.choice(population, p=weights)


def crossover(parent1, parent2):
    c1 = np.random.randint(0, n, dtype=dt)
    c2 = np.random.randint(c1 + 1, n + 1, dtype=dt)

    child = np.full(n, n, dtype=dt)
    
    child[c1:c2] = parent1[c1:c2]

    values = np.sort(child[c1:c2])
    values_size = len(values)
    
    j = 0
    for i in range(c1):
        while True:
            index = np.searchsorted(values, parent2[j])
            if index == values_size or (values[index] != parent2[j]):
                break
            else:
                j += 1

        child[i] = parent2[j]
        j += 1

    for i in range(c2, len(parent2)):
        while True: 
            index = np.searchsorted(values, parent2[j])
            if index == values_size or (values[index] != parent2[j]):
                break
            else:
                j += 1
        
        child[i] = parent2[j]
        j += 1

    return child


def mutation(individual):
    i = np.random.randint(0, n, dtype=dt)
    j = np.random.randint(0, n, dtype=dt)

    individual[i], individual[j] = individual[j], individual[i]

    return individual


def main():
    start = time.time()

    population = initialize()

    population_fitness = np.array([fitness(organism) for organism in population], dtype=fdt)

    solution = [0, 0]
    original = 0

    for i in range(generations):
        print(i)

        indices = population_fitness.argsort()

        population_fitness = population_fitness[indices]
        population = population[indices]

        if i == 0 or population_fitness[0] < solution[0]:
            solution = [population_fitness[0], population[0]]

        if i == 0:
            original = population[0]

        if population_fitness[-1] - population_fitness[0] < 1:
            print("CONVERGED")
            break

        print(population_fitness[0])
        #print(solution[0])

        max_value = population_fitness[-1]
        total_sum = max_value * population_size - np.sum(population_fitness)
        weights = np.array([(max_value - x) / total_sum for x in population_fitness], dtype=fdt)

        new_population = np.zeros(shape=(population_size, n), dtype=dt)
        new_population_fitness = np.zeros(population_size, dtype=fdt)

        for j in range(population_size):
            parent1 = selection(population, weights)
            parent2 = selection(population, weights)

            if np.random.rand() <= crossover_probability:
                child = crossover(parent1, parent2)
            else:
                child = rng.choice(np.array([parent1, parent2]))

            if np.random.rand() <= mutation_probability:
                child = mutation(child)

            new_population[j] = child
            new_population_fitness[j] = fitness(child)

        elite_population_size = int(elitism * population_size)
        
        normal = rng.choice(np.arange(population_size, dtype=dt), population_size - elite_population_size)

        population[int(elitism * population_size):] = new_population[normal]
        population_fitness[int(elitism * population_size):] = new_population_fitness[normal]
        

    print(str(time.time() - start) + "s")

    print(original)
    print(solution[0])
    
    solution_points = np.array([city_map[x] for x in solution[1]] + [city_map[solution[1][0]]])
    x = solution_points[:, 0]
    y = solution_points[:, 1]

    plt.figure()
    plt.text(1, 1, solution[0])
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
