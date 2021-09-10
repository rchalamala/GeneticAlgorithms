#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:34:40 2021

@author: Rahul Chalamala
"""

import numpy as np
import copy

'''
seed = 0

np.random.seed(seed)
rng = np.random.default_rng(seed)
'''

rng = np.random.default_rng()


population_size = 500
generations = 10000
crossover_probability = 0.7
mutation_probability = 0.6
elitism = 0.7


n = 9
blocksize = 3

assert n % blocksize == 0


def process(board, solution):
    combined = copy.deepcopy(board)

    for row in range(n):
        replaced = 0

        for column in range(n):
            if combined[row][column] == 0:
                combined[row][column] = solution[row][replaced]
                replaced += 1

    return combined


def output(board):
    for row in board:
        for point in row:
            print(point, end='')
        print()
    print()


def generate_missing(goal):
    return np.array([np.array(list(set(np.arange(1, n + 1)).difference(set(goal[row])))) for row in range(n)], dtype=object)


'''
def generate_bounds(goal, missing):
    bounds = []
    for row in range(n):
        bound = [0]
        total = 0
        for column in range(n):
            if column and column % blocksize == 0:
                bound.append(total)
            if goal[row, column] == 0:
                total += 1
        bound.append(total)
        bounds.append(np.array(bound))
    return np.array(bounds, dtype=object)
'''


def presolve(board):
    rows = [set(np.arange(1, n + 1)) for _ in range(n)]
    columns = [set(np.arange(1, n + 1)) for _ in range(n)]
    blocks = [set(np.arange(1, n + 1)) for _ in range(n)]

    for row in range(n):
        for column in range(n):
            if board[row, column] > 0:
                rows[row].remove(board[row, column])
                columns[column].remove(board[row, column])
                blocks[3 * (row // 3) + column // 3].remove(board[row, column])

    row = 0

    while row < n:
        column = 0

        while column < n:
            block = 3 * (row // 3) + column // 3

            available = rows[row].union(columns[column].union(blocks[block]))

            if len(available) == 1:
                item = list(available[0])

                board[row, column] = item

                rows[row].remove(item)
                columns[column].remove(item)
                blocks[block].remove(item)

                break

            column += 1

        if column < n:
            row = 0

        else:
            row += 1


def initialize(missing):
    population = []

    for _ in range(population_size):
        individual = copy.deepcopy(missing)

        for row in range(n):
            rng.shuffle(individual[row])

        population.append(individual)

    return population


def fitness(solution, goal):
    combined = copy.deepcopy(goal)

    for row in range(n):
        replaced = 0

        for column in range(n):
            if combined[row][column] == 0:
                combined[row][column] = solution[row][replaced]
                replaced += 1

    total = 0

    for iterator in range(n):
        total += 2 * n - len(set(combined[iterator])) - len(set(combined[:, iterator]))

    for row in range(blocksize):
        for column in range(blocksize):
            total += blocksize * blocksize - len(set(combined[blocksize * row:blocksize * (row + 1), blocksize * column:blocksize * (column + 1)].flatten()))

    return total


def selection(population, weights):
    return rng.choice(population, p=weights)


'''
def crossover(parent1, parent2, bounds):
    parent1Blocks = rng.choice(np.arange(0, n), np.random.randint(1, n))

    if np.random.randint(0, 2):
        parent1, parent2 = parent2, parent1

    child1 = copy.deepcopy(parent2)
    child2 = copy.deepcopy(parent1)

    for block in parent1Blocks:
        for row in range(blocksize):
            here = blocksize * (block // blocksize) + row
            column = block % blocksize

            child1[here][bounds[here][column]:bounds[here][column + 1]] = parent1[here][bounds[here][column]:bounds[here][column + 1]]
            child2[here][bounds[here][column]:bounds[here][column + 1]] = parent2[here][bounds[here][column]:bounds[here][column + 1]]

    return [child1, child2]


def mutation(individual, bounds):
    row = np.random.randint(0, n)
    
    mutant = copy.deepcopy(individual)
    
    swap = np.random.randint(0, len(bounds[row]) - 1)

    rng.shuffle(mutant[row][bounds[row][swap]:bounds[row][swap + 1]])

    return mutant
'''


def crossover(parent1, parent2):
    c1 = np.random.randint(0, n)
    c2 = np.random.randint(c1 + 1, n + 1)

    child = copy.deepcopy(parent1)

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

    # '''
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
    goal = np.array([np.array([1,0,0,0,0,7,0,9,0]),
                     np.array([0,3,0,2,0,0,0,0,8]),
                     np.array([0,0,9,6,0,0,5,0,0]),
                     np.array([0,0,5,3,0,0,9,0,0]),
                     np.array([0,1,0,0,8,0,0,0,2]),
                     np.array([6,0,0,0,0,4,0,0,0]),
                     np.array([3,0,0,0,0,0,0,1,0]),
                     np.array([0,4,0,0,0,0,0,0,7]),
                     np.array([0,0,7,0,0,0,3,0,0])], dtype=object)
    '''

    presolve(goal)

    missing = generate_missing(goal)

    # bounds = generate_bounds(goal, missing)

    population = initialize(missing)

    solution = [0, np.empty(n)]

    for i in range(generations):
        print(i)

        population_fitness = [fitness(x, goal) for x in population]

        best_fitness = min(population_fitness)
        best_index = population_fitness.index(best_fitness)
        best_individual = population[best_index]

        if i == 0 or best_fitness < solution[0]:
            solution = [best_fitness, best_individual]

        if max(population_fitness) == min(population_fitness) or solution[0] == 0:
            if solution[0] == 0:
                print("CONVERGED")

                break

            else:
                print("CONVERGED AND RESTARTING")
                
                population = initialize(missing)
                i = -1

                continue

        print(max(population_fitness))
        print(min(population_fitness))
        print(solution[0])

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

            population.append(child)
            population_fitness.append(fitness(child, goal))

        population = [x for _, x in sorted(zip(population_fitness, population), key=lambda pair: pair[0])]

        elite = population[:int(elitism * population_size)]
        normal = rng.choice(np.array(population[int(elitism * population_size):], dtype=object), population_size - int(elitism * population_size)).tolist()

        population = elite
        population.extend(normal)

    output(goal)
    output(process(goal, solution[1]))

    print(solution[0])


if __name__ == "__main__":
    main()
