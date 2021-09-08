#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:30:31 2021

@author: Rahul Chalamala
"""

import numpy as np
import matplotlib.pyplot as plt

'''
seed = 0

np.random.seed(seed)
rng = np.random.default_rng(seed)
'''

rng = np.random.default_rng()


lb = -6
ub = 6

population_size = 500
generations = 1000
crossover_probability = 0.65
mutation_probability = 0.85
elitism = True


n = 2


def f(x):
    return x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1]) + 20 # rastrigin
    # return np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2))
    # return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2) # easom
    # return x[0] ** 2 + x[1] ** 2


if n == 1:
    x = np.linspace(lb, ub, 100000)
    y = np.array([f([i]) for i in x])

    plt.figure()
    plt.plot(x, y)
elif n == 2:
    basex = np.linspace(lb, ub, 1000)
    basey = np.linspace(lb, ub, 1000)

    x, y = np.meshgrid(basex, basey)
    z = f([x, y])

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')


def initialize():
    return list(lb + np.random.rand(population_size, n) * (ub - lb))


def fitness(solution):
    return f(solution)


def selection(population, weights):
    return rng.choice(population, p=weights)


def crossover(parent1, parent2):
    child = np.empty(n)

    for axis in range(n):
        child[axis] = parent1[axis] + np.random.rand() * (parent2[axis] - parent1[axis])

    return child


def mutation(individual):
    mutant = np.empty(n)

    for axis in range(n):
        mutant[axis] = rng.normal(individual[axis], (ub - lb) / 2, 1)

    return mutant


def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def main():
    population = initialize()
    solution = [0, np.empty(n)]

    for i in range(generations):
        print(i)

        population_fitness = [fitness(x) for x in population]

        best_fitness = min(population_fitness)
        best_index = population_fitness.index(best_fitness)
        best_individual = population[best_index]

        if i == 0 or best_fitness < solution[0]:
            solution = [best_fitness, best_individual]

        if max(population_fitness) - min(population_fitness) < 1e-7:
            print("CONVERGED")
            break

        print(max(population_fitness))
        print(min(population_fitness))
        print(solution[0])

        new_population = []

        max_value = max(population_fitness)
        total_sum = max_value * population_size - sum(population_fitness)
        weights = np.array([(max_value - x) / total_sum for x in population_fitness])
        weights /= weights.sum()

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

        if elitism:
            population.extend(new_population)

            population_fitness.extend([fitness(x) for x in new_population])

            population = [x for _, x in sorted(zip(population_fitness, population), key=lambda pair : pair[0])][:population_size]
        else:
            population = new_population

    point = tuple(flatten((tuple(solution[1]), fitness(solution[1]))))

    if n == 1:
        plt.annotate('(%s, %s)' % point, xy=point, textcoords='data')
    elif n == 2:
        ax.text(solution[1][0], solution[1][1], fitness(solution[1]), '(%s, %s, %s)' % point, zorder=1)
        ax.scatter(solution[1][0], solution[1][1], fitness(solution[1]), color='black')

    print(point)


if __name__ == "__main__":
    main()
