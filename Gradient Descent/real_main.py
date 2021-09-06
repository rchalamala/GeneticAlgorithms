#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:30:31 2021

@author: Rahul Chalamala
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import gmpy2
from gmpy2 import xmpz

'''
seed = 0

np.random.seed(seed)
rng = np.random.default_rng(seed)
'''

rng = np.random.default_rng()


lb = -10
ub = 10

population_size = 1000
generations = 10
crossover_probability = 0.75
mutation_probability = 0.3


n = 2


def f(x):
    return np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2))
    return -np.cos(np.array([x[0]])) * np.cos(np.array([x[1]])) * np.exp(np.array([- (x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2]))
    return x[0] ** 2 + x[1] ** 2


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
    return lb + np.random.rand(population_size, n) * (ub - lb)


def fitness(solution):
    return f(solution).item()


def selection(population, weights):
    return rng.choice(population, p=weights)


def crossover(parent1, parent2):
    child = np.empty(n)

    for axis in range(n):
        child[axis] = (parent1[axis] + parent2[axis]) / 2

    return child


def mutation(individual):

    mutant = np.empty(n)

    for axis in range(n):
        mutant[axis] = rng.normal(individual[axis], (ub - lb) / 10, 1)

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

        if max(population_fitness) - min(population_fitness) < 1e-10:
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

            if np.random.random() <= crossover_probability:
                child = crossover(parent1, parent2)
            else:
                child = rng.choice(np.array([parent1, parent2]))

            if np.random.random() < mutation_probability:
                child = mutation(parent1)

            new_population.append(child)

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
