import numpy as np
from deap import base
from deap import creator
from deap import tools


class GenAttack:
    def __init__(self, dist_delta, step_size, pop_size, cx_prob, mut_prob, model):
        self.dist_delta = dist_delta
        self.step_size = step_size
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.model = model

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

        def evaluate(individual):
            # TODO: do stuff to get model prediction
            target_prediction = sum(individual)  # just for demonstration
            other_prediction = 1  # max prediction of other label != target

            return (np.log10(target_prediction) - np.log10(other_prediction),)

        # individual of random values
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              distribution, n=2 ** 2)
        # population of random individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # evaluation function
        self.toolbox.register("evaluate", evaluate)

    def attack(self, orig_img, target, pop_size, num_gen):
        # initialize population
        pop = self.toolbox.population(n=pop_size)

        for ind in pop:
            print(ind)

        for g in range(num_gen):
            # evaluate the entire population
            fitnesses = map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                print("fit=", fit)
                ind.fitness.values = fit

            # check if best individual == target (keep best individual)

            # select the next generation individuals
            offspring = list(map(self.toolbox.clone, tools.selRoulette(pop, len(pop))))

            # mate the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.cx_prob:
                    # TODO: uniform crossover (f(child1) / (f(child1) + f(child2))
                    del child1.fitness.values
                    del child2.fitness.values

            # mutate the offspring
            for mutant in offspring:
                if np.random.random() < self.mut_prob:
                    # TODO: mutate (bernoulli(p) * uniform(-delta, delta)
                    del mutant.fitness.values

            # replace population with new offspring
            pop[:] = offspring
