# -*- coding = utf-8 -*-
# @Time :  9:52
# @Author : cjj
# @File : GA_Second_best.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt

POP_SIZE = 200
MUTATION_RATE = 0.1
N_GENERATIONS = 20
X_BOUND = [0, 5]

def F(x):
    return np.sin(10 * x) * x + np.cos( 2 * x) * x

def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)

def select(pop, fitness):
    index = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=fitness/fitness.sum())
    return pop[index]

def crossover(pop):
    for index in range(len(pop)):
        if index % 2 == 0:
            pop[index,] = (pop[index,] + pop[index + 1,]) / 2
            pop[index + 1,] = (pop[index,] + pop[index + 1,]) / 2 + 0.01
    return pop

def mutate(pop):
    for index in range(len(pop)):
        if np.random.rand() < MUTATION_RATE:
            pop[index,] = np.random.rand()*X_BOUND[1]
    return pop


x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))
pop = np.sort(np.random.rand(POP_SIZE) * X_BOUND[1])
for i in range(N_GENERATIONS):

    fitness = get_fitness(F(pop))
    pop = select(pop, fitness)
    pop = np.sort(pop)

    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(pop, F(pop), s=200, lw=0, c='red', alpha=0.5); plt.pause(0.2)

    pop = crossover(pop)
    pop = mutate(pop)
    print("Most fitted sample: ", pop[np.argmax(fitness)])

plt.ioff(),plt.show()




