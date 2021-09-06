#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:30:31 2021

@author: Rahul Chalamala
"""

import numpy as np
import matplotlib.pyplot as plt
import gmpy2
from gmpy2 import xmpz


rng = np.random.default_rng()


lb = -100
ub = 100

population_size = 150
generations = 100
crossover_probability = 0.65
mutation_probability = 0.3


n = 2


def f(x):
    return (-np.cos(np.array([x[0]])) * np.cos(np.array([x[1]])) * np.exp(np.array([- (x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2]))).item()


'''
x = np.linspace(-1000, 1000, 100000)
y = np.array([f([i]) for i in x])

plt.figure()
plt.plot(x, y)
'''


def initialize():
    return np.random.randint(lb, ub, (population_size, n))


def fitness(solution):
    return f(solution)


def selection(population, weights):
    return rng.choice(population, p=weights)


def crossover(parent1, parent2):
    child = np.empty(n)

    for axis in range(n):
        signs = np.array([1 if parent1[axis] >= 0 else -1, 1 if parent2[axis] >= 0 else -1])

        bits1 = xmpz(int(parent1[axis] * signs[0]))
        bits2 = xmpz(int(parent2[axis] * signs[1]))

        if len(bits1) > len(bits2):
            bits1, bits2 = bits2, bits1

        c1 = np.random.randint(0, len(bits1))
        c2 = np.random.randint(c1 + 1, len(bits1) + 1)

        crossed = xmpz(0)

        crossed[c1:c2] = bits1[c1:c2]

        crossed[:c1] = bits2[:c1]
        crossed[c2:] = bits2[c2:]

        child[axis] = rng.choice(signs) * int(crossed)

    return child


def mutation(individual):
    mutant = np.empty(n)

    for axis in range(n):
        bits = xmpz(int(individual[axis] * (1 if individual[axis] >= 0 else -1)))

        for bit in range(len(bits)):
            if np.random.randint(0, 2) == 1:
                bits[bit] ^= 1

        mutant[axis] = int(bits) * (1 if np.random.randint(0, 2) == 0 else -1)

    return mutant


def main():
    population = initialize()
    solution = [0, np.empty(n)]

    for i in range(generations):
        print(i)

        population_fitness = [fitness(x) for x in population]

        if max(population_fitness) == min(population_fitness):
            print("CONVERGED")
            break

        best_fitness = min(population_fitness)
        best_index = population_fitness.index(best_fitness)
        best_individual = population[best_index]

        if i == 0 or best_fitness < solution[0]:
            solution = [best_fitness, best_individual]

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

    point = (solution[1], solution[0])

    # plt.annotate('(%s, %s)' % point, xy=point, textcoords='data')

    print(point)


if __name__ == "__main__":
    main()
