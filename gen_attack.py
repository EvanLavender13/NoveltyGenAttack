from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools


class GenAttack:
    def __init__(self, model, dist_delta=0.3, step_size=1, mut_prob=0.05):
        self.dist_delta = dist_delta
        self.step_size = step_size
        self.mut_prob = mut_prob
        self.model = model
        self.image = None
        self.orig_index = 0
        self.target_index = 0
        self.stop = False
        self.evaluations = 0
        self.adversarial_image = None

        self.plot_img = None

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
            ind = np.array(individual)
            ind = ind.reshape((28, 28))
            ind = self.image + ind
            self.plot_img.set_data(ind)
            plt.draw()
            plt.pause(0.00001)

            img = (np.expand_dims(ind, 0))
            predictions = self.model.predict(img)[0]
            target_prediction = predictions[self.target_index]
            if np.argmax(predictions) == self.target_index:
                self.stop = True
                self.adversarial_image = ind
                #print("bazinga!")

            predictions[self.target_index] = -999

            other_prediction_index = np.argmax(predictions)
            other_prediction = predictions[other_prediction_index]

            #print("target_prediction=", self.target_index, target_prediction)
            #print("other_prediction=", other_prediction_index, other_prediction)

            self.evaluations += 1
            #print("evaluations=", self.evaluations)

            return (np.log10(target_prediction) - np.log10(other_prediction) + 1,)

        self.toolbox.register("random_distribution", distribution)
        # individual of random values
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.random_distribution, n=28 ** 2)
        # population of random individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # evaluation function
        self.toolbox.register("evaluate", evaluate)

    def attack(self, orig_img, orig_index, target_index, pop_size, num_gen):
        self.image = orig_img
        self.orig_index = orig_index
        self.target_index = target_index

        # initialize population
        pop = self.toolbox.population(n=pop_size)

        plt.figure()
        plt.ion()
        plt.show()

        img = np.array(max(pop))
        img = img.reshape((28, 28))
        self.plot_img = plt.imshow(orig_img + img)
        plt.draw()
        plt.pause(0.001)

        for g in range(num_gen):
            # evaluate the entire population
            fitnesses = map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # check if best individual == target (keep best individual)
            # print("evaluations=", self.evaluations)
            if self.stop:
                print("finished after", self.evaluations, "evaluations")
                break

            # normalize for negative fitness values
            fits = [ind.fitness.values[0] for ind in pop]
            min_fit = abs(min(fits)) + 1
            for ind in pop:
                ind.fitness.values += min_fit

            # select the next generation individuals
            offspring = list(map(self.toolbox.clone, tools.selRoulette(pop, len(pop))))

            # mate the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                prob = child1.fitness.values[0] / (child1.fitness.values[0] + child2.fitness.values[0])
                tools.cxUniform(child1, child2, prob)
                del child1.fitness.values
                del child2.fitness.values

            # mutate the offspring
            for mutant in offspring:
                for x in range(len(mutant)):
                    mutant[x] += self.toolbox.random_distribution()

                del mutant.fitness.values

            # replace population with new offspring
            pop[:] = offspring
            # add elite member
