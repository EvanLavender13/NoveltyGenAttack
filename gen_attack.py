import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools
import multiprocessing

class GenAttack:
    def __init__(self, model, dist_delta=0.3, step_size=1, mut_prob=0.05):
        self.dist_delta = dist_delta
        self.step_size = step_size
        self.mut_prob = mut_prob
        self.model = model
        self.image = None
        self.target_index = 0
        self.stop = False
        self.evaluations = 0
        self.evaluation_found = 0
        self.adversarial_image = None

        self.plot_img = None

        self.toolbox = base.Toolbox()

        np.random.seed(64)

        pool = multiprocessing.Pool()
        self.toolbox.register("map", pool.map)

        # create a maximizing fitness value
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # create an individual that is a list of values to be added to image
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.toolbox.register("random_distribution", self.distribution)
        # individual of random values
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.random_distribution, n=28 ** 2)
        # population of random individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # evaluation function
        self.toolbox.register("evaluate", self.evaluate)

    def evaluate(self, individual):
        """
        Evaluates an individual by classifying it with the given model.

        Args:
            individual: individual to be evaluated, an image in this case

        Returns:
            log(target prediction) - log(max prediction != target)
        """
        ind = np.array(individual)
        ind = ind.reshape((28, 28, 1))

        image = (np.expand_dims(ind, 0))
        predictions = self.model.predict(image)[0]
        self.evaluations += 1
        target_prediction = predictions[self.target_index]
        if np.argmax(predictions) == self.target_index:
            self.stop = True
            self.evaluation_found = self.evaluations

        predictions[self.target_index] = -999

        other_prediction_index = np.argmax(predictions)
        other_prediction = predictions[other_prediction_index]

        #print("target_prediction=", target_prediction)
        #print("other_prediction=", other_prediction)

        if self.evaluations % 500 == 0:
            print("eval:", self.evaluations)

        return (np.log10(target_prediction) - np.log10(other_prediction),)

    def distribution(self):
        """
        A bernoulli distribution in the given range of distortion

        Returns:
            Bernoulli(p) * U(-dist, dist)
        """
        return np.random.binomial(n=1, p=self.mut_prob) * np.random.uniform(low=-self.dist_delta,
                                                                            high=self.dist_delta)

    def attack(self, image, target_index, pop_size, num_eval=100000, draw=False):
        self.image = image
        self.target_index = target_index
        self.stop = False
        self.evaluations = 0
        self.evaluation_found = 0

        # initialize population
        pop = self.toolbox.population(n=pop_size)
        flattened_image = image.flatten()
        images = [flattened_image] * pop_size

        for individual, image in zip(pop, images):
            for i in range(len(individual)):
                individual[i] += image[i]

        if draw:
            plt.figure()
            plt.ion()
            plt.show()
            img = np.array(pop[0])
            img = img.reshape((28, 28))
            self.plot_img = plt.imshow(img)
            plt.draw()
            plt.pause(0.001)

        while self.evaluations < num_eval:
            # evaluate the entire population
            fitnesses = map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # check if best individual == target (keep best individual)
            # print("evaluations=", self.evaluations)
            if self.stop:
                return self.evaluation_found

            # normalize for negative fitness values
            fits = [ind.fitness.values[0] for ind in pop]
            min_fit = abs(min(fits)) + 1
            for ind in pop:
                ind.fitness.values += min_fit

            # select the next generation individuals
            # offspring = list(map(self.toolbox.clone, tools.selRoulette(pop, len(pop))))
            offspring = []
            max = sum(ind.fitness.values[0] for ind in pop)
            selection_probs = [ind.fitness.values[0] / max for ind in pop]
            while len(offspring) < pop_size:
                parent1 = pop[np.random.choice(pop_size, p=selection_probs)]
                parent2 = pop[np.random.choice(pop_size, p=selection_probs)]
                child = self.toolbox.individual()

                prob = parent1.fitness.values[0] / (parent1.fitness.values[0] + parent2.fitness.values[0])
                for i in range(len(child)):
                    parent = parent1 if np.random.random() < prob else parent2
                    child[i] = parent[i]

                    child[i] += self.toolbox.random_distribution()
                    del child.fitness.values

                offspring.append(child)

            # replace population with new offspring
            pop[:] = offspring
            # add elite member

            # perform image clipping
            for ind in pop:
                for i in range(len(ind)):
                    if ind[i] - flattened_image[i] > self.dist_delta:
                        ind[i] = flattened_image[i] + self.dist_delta
                    elif abs(ind[i] - flattened_image[i]) > self.dist_delta:
                        ind[i] = flattened_image[i] - self.dist_delta

            if draw:
                ind = np.array(pop[0])
                ind = ind.reshape((28, 28))
                self.plot_img.set_data(ind)
                plt.draw()
                plt.pause(0.00001)

        return num_eval
