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
import os
import imageio

rng = np.random.default_rng()


n = 30

lb = 0
ub = 1000

population_size = 1000
generations = 1000
crossover_probability = 0.6
elitism = 0.1
# mutation_probability = 1 - crossover_probability - elitism


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


def mutation1(individual):
    c1 = np.random.randint(0, n, dtype=dt)
    c2 = np.random.randint(c1 + 1, n + 1, dtype=dt)

    individual[c1:c2] = individual[c1:c2][::-1]

    return individual

    i = np.random.randint(0, n, dtype=dt)
    j = np.random.randint(0, n, dtype=dt)

    individual[i], individual[j] = individual[j], individual[i]

    return individual


def mutation2(individual):
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

    last_time = 0

    update_count = 0

    #for i in range(generations):
    i = 0
    
    while i < generations or last_time < 100:
        print(i, last_time, update_count)

        indices = population_fitness.argsort()

        population_fitness = population_fitness[indices]
        population = population[indices]

        if i == 0 or population_fitness[0] < solution[0]:
            solution = [population_fitness[0], population[0]]
            solution_points = np.array([city_map[x] for x in solution[1]] + [city_map[solution[1][0]]])

            x = solution_points[:, 0]
            y = solution_points[:, 1]

            plt.figure()
            plt.text(1, 1, f"{update_count} {solution[0]}")
            plt.scatter(x, y)
            plt.plot(x, y)

            plt.savefig(f"images/{update_count}_{os.path.basename(__file__[:-3])}.png")
            plt.close()

            print(f"Updated: {i}, {last_time}, {update_count}")

            update_count += 1

            last_time = 0

        if i == 0:
            original = population[0]

        if population_fitness[-1] - population_fitness[0] < 1e-1:
            print("CONVERGED")
            break

        print(population_fitness[0])
        #print(solution[0])

        first_bound = int(elitism * population_size)
        second_bound = int((elitism + crossover_probability) * population_size)

        max_value = population_fitness[second_bound - 1]
        total_sum = max_value * (second_bound - first_bound) - np.sum(population_fitness[first_bound:second_bound])
        weights = np.array([(max_value - x) / total_sum for x in population_fitness[first_bound:second_bound]], dtype=fdt)

        for j in range(second_bound - first_bound):
            parent1 = selection(population[first_bound:second_bound], weights)
            parent2 = selection(population[first_bound:second_bound], weights)

            child = crossover(parent1, parent2)

            population[first_bound + j] = child
            population_fitness[first_bound + j] = fitness(child)
        
        for j in range(population_size - second_bound - first_bound):
            if np.random.rand() <= 0.5:
                population[second_bound + j] = mutation1(population[second_bound + j])
            else:
                population[second_bound + j] = mutation2(population[second_bound + j])

            population_fitness[second_bound + j] = fitness(population[second_bound + j])

        i += 1
        last_time += 1

        if i % 100 == 0:
            with imageio.get_writer(f"gifs/tsp_{i // 100}_{os.path.basename(__file__[:-3])}.gif", mode='I') as writer:
                for j in range(update_count):
                    image = imageio.v2.imread(f"images/{j}_{os.path.basename(__file__[:-3])}.png")
                    for _ in range(1):
                        writer.append_data(image)
                    if j == update_count - 1:
                        for _ in range(20):
                            writer.append_data(image)
        

    print(str(time.time() - start) + "s")

    print(original)
    print(solution[0])

    with imageio.get_writer(f"gifs/tsp_{i // 100}_{os.path.basename(__file__[:-3])}.gif", mode='I') as writer:
        for j in range(update_count):
            image = imageio.v2.imread(f"images/{j}_{os.path.basename(__file__[:-3])}.png")
            for _ in range(1):
                writer.append_data(image)
            if j == update_count - 1:
                for _ in range(20):
                    writer.append_data(image)
        os.remove(f"images/{j}_{os.path.basename(__file__[:-3])}.png")


if __name__ == "__main__":
    main()
