import numpy as np
from deap import base
from deap import creator
from deap import tools

from novelty_eval import NoveltyEvaluator


class NoveltyAttack:
    def __init__(self, model, dist_delta=0.3, step_size=1, pop_size=6, mut_prob=0.05):
        self.dist_delta = dist_delta
        self.step_size = step_size
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.model = model

        self.toolbox = base.Toolbox()

        self.pop = []
        self.novelty_eval = NoveltyEvaluator(pop_size=self.pop_size)

        np.random.seed(64)

        # create a maximizing fitness value
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # create an individual that is a list of values to be added to image
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # random distribution - bernoulli(p) * U(-delta, delta)
        def distribution():
            return np.random.binomial(n=1, p=self.mut_prob) * np.random.uniform(low=-self.dist_delta,
                                                                                high=self.dist_delta)

        def evaluate(individual):
            # TODO: do stuff to get model prediction

            behavior = []  # get predictions from model

            # evaluate novelty of behavior
            novelty = self.novelty_eval.evaluate_novelty(self.pop, behavior)

            return novelty

        # individual of random values
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              distribution, n=2 ** 2)
        # population of random individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # evaluation function
        self.toolbox.register("evaluate", evaluate)

    def attack(self, orig_img, target, num_gen):
        # initialize population
        self.pop = self.toolbox.population(n=self.pop_size)

        for ind in self.pop:
            print(ind)

        for g in range(num_gen):
            # evaluate the entire population
            fitnesses = map(self.toolbox.evaluate, self.pop)
            for ind, fit in zip(self.pop, fitnesses):
                print("fit=", fit)
                ind.fitness.values = fit

            # check if best individual == target (keep best individual)

            # select the next generation individuals
            offspring = list(map(self.toolbox.clone, tools.selRoulette(self.pop, len(self.pop) - 1)))

            # mate the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # TODO: uniform crossover (f(child1) / (f(child1) + f(child2))
                del child1.fitness.values
                del child2.fitness.values

            # mutate the offspring
            for mutant in offspring:
                if np.random.random() < self.mut_prob:
                    # TODO: mutate (bernoulli(p) * uniform(-delta, delta)
                    del mutant.fitness.values

            # replace population with new offspring
            self.pop[:] = offspring
            # add elite member
