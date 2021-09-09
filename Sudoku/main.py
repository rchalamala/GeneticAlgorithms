#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:34:40 2021

@author: Rahul Chalamala
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

'''
seed = 0

np.random.seed(seed)
rng = np.random.default_rng(seed)
'''

rng = np.random.default_rng()


population_size = 1000
generations = 10000
crossover_probability = 0.95
mutation_probability = 0.85
elitism = 0.3


n = 9
blocks = 3

assert n % blocks == 0

blocksize = n // blocks

'''
goal = np.array([np.array([4,0,1,2,9,0,0,7,5]),
                 np.array([2,0,0,3,0,0,8,0,0]),
                 np.array([0,7,0,0,8,0,0,0,6]),
                 np.array([0,0,0,1,0,3,0,6,2]),
                 np.array([1,0,5,0,0,0,4,0,3]),
                 np.array([7,3,0,6,0,8,0,0,0]),
                 np.array([6,0,0,0,2,0,0,3,0]),
                 np.array([0,0,7,0,0,1,0,0,4]),
                 np.array([8,9,0,0,6,5,1,0,7])], dtype=object)
'''
#'''
goal = np.array([np.array([1,0,0,0,0,7,0,9,0]),
                 np.array([0,3,0,2,0,0,0,0,9]),
                 np.array([0,0,9,6,0,0,5,0,0]),
                 np.array([0,0,5,3,0,0,9,0,0]),
                 np.array([0,1,0,0,8,0,0,0,2]),
                 np.array([6,0,0,0,0,4,0,0,0]),
                 np.array([3,0,0,0,0,0,0,1,0]),
                 np.array([0,4,0,0,0,0,0,0,7]),
                 np.array([0,0,7,0,0,0,3,0,0])], dtype=object)
#'''

missing = []

for row in range(n):
    used = set(goal[row])
    length = n - len(used) + (0 in used)
    number = 1
    values = []
    while length:
        if number not in used:
            length -= 1
            values.append(number)
        number += 1
    missing.append(np.array(values))

missing = np.array(missing, dtype=object)


def initialize(size):
    population = []

    for _ in range(size):
        individual = copy.deepcopy(missing)
        for row in range(n):    
            rng.shuffle(individual[row])
        population.append(individual)

    return population


def fitness(solution):
    combined = copy.deepcopy(goal)
    for row in range(n):
        replaced = 0
        for column in range(n):
            if combined[row][column] == 0:
                combined[row][column] = solution[row][replaced]
                replaced += 1

    total = 0

    for iterator in range(n):
        total += 2 * n - len(set(combined[iterator])) - len(set(combined[:,iterator]))

    for row in range(blocks):
        for column in range(blocks):
            total += blocksize * blocksize - len(set(combined[blocksize * row:blocksize * (row + 1), blocksize * column:blocksize * (column + 1 )].flatten()))

    return total


def selection(population, weights):
    return rng.choice(population, p=weights)


def crossover(parent1, parent2):
    c1 = np.random.randint(0, n)
    c2 = np.random.randint(c1 + 1, n + 1)

    child = copy.deepcopy(missing)

    child[:c1] = parent2[:c1]

    child[c1:c2] = parent1[c1:c2]

    child[c2:] = parent2[c2:]

    return child


def mutation(individual):
    row = np.random.randint(0, n)

    c1 = np.random.randint(0, len(individual[row]))
    c2 = np.random.randint(0, len(individual[row]))

    mutant = copy.deepcopy(individual)

    mutant[row][c1], mutant[row][c2] = mutant[row][c2], mutant[row][c1]

    return mutant


def main():
    population = initialize(population_size)
    solution = [0, np.empty(n)]

    for i in range(generations):
        print(i)

        population_fitness = [fitness(x) for x in population]

        best_fitness = min(population_fitness)
        best_index = population_fitness.index(best_fitness)
        best_individual = population[best_index]

        if i == 0 or best_fitness < solution[0]:
            solution = [best_fitness, best_individual]

        if max(population_fitness) - min(population_fitness) < 1e-7 or solution[0] == 0:
            if solution[0] == 0:
                print("CONVERGED")
                break
            else:
                print("CONVERGED AND RESTARTING")
                
                population = initialize(population_size)
                
                i -= 1
                continue

        print(max(population_fitness))
        print(min(population_fitness))
        print(solution[0])

        new_population = []

        max_value = max(population_fitness)
        total_sum = max_value * population_size - sum(population_fitness)
        weights = np.array([(max_value - x) / total_sum for x in population_fitness])
        weights /= weights.sum()
        
        nppopulation = np.array(population, dtype=object)

        for j in range(population_size):
            parent1 = selection(nppopulation, weights)
            parent2 = selection(nppopulation, weights)


            if np.random.rand() <= crossover_probability:
                child = crossover(parent1, parent2)
            else:
                child = rng.choice(np.array([parent1, parent2]))

            if np.random.rand() <= mutation_probability:
                child = mutation(child)

            new_population.append(child)

        population.extend(new_population)

        population_fitness.extend([fitness(x) for x in new_population])

        population = [x for _, x in sorted(zip(population_fitness, population), key=lambda pair: pair[0])]

        elite = population[:int(elitism * population_size)]
        normal = rng.choice(np.array(population[int(elitism * population_size):], dtype=object), population_size - int(elitism * population_size)).tolist()
        
        population = elite
        population.extend(normal)

    for row in goal:
        for point in row:
            print(point, end=' ')
        print()
    print()

    combined = copy.deepcopy(goal)
    for row in range(n):
        replaced = 0
        for column in range(n):
            if combined[row][column] == 0:
                combined[row][column] = solution[1][row][replaced]
                replaced += 1

    for row in combined:
        for point in row:
            print(point, end=' ')
        print()
    print()
    
    print(solution[0])


if __name__ == "__main__":
    main()
