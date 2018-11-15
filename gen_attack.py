import numpy as np
from deap import base
from deap import creator
from deap import tools


class GenAttack:
    def __init__(self, dist_delta, step_size, pop_size, cx_prob, mut_prob):
        self.dist_delta = dist_delta
        self.step_size = step_size
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob

        self.toolbox = base.Toolbox()

        np.random.seed(64)

        # create a maximizing fitness value
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # create an individual that is a list of values to be added to image
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # random distribution - bernoulli(p) * U(-delta, delta)
        def distribution():
            return np.random.binomial(n=1, p=self.mut_prob) * np.random.uniform(low=-self.dist_delta,
                                                                                high=self.dist_delta)

        # individual of random values
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              distribution, n=2 ** 2)
        # population of random individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def attack(self, orig_img, target, model, num_gen):
        # initialize population
        pop = self.toolbox.population(n=25)

        for ind in pop:
            print(ind)

        for g in range(num_gen):
            continue
            # compute fitnesss

            # check if best individual == target

            # roulette selection

            # uniform crossover

            # mutation
