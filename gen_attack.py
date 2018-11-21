import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools
import multiprocessing

class GenAttack:
    def __init__(self, model, image_shape, image_dim, dist_delta=0.3, step_size=1, mut_prob=0.05):
        self.dist_delta = dist_delta
        self.step_size = step_size
        self.mut_prob = mut_prob
        self.model = model
        self.image_shape = image_shape
        self.image_dim = image_dim
        self.image = None
        self.index = 0
        self.stop = False
        self.evaluations = 0
        self.evaluation_found = 0
        self.adversarial_image = None
        self.targeted = True

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
                              self.toolbox.random_distribution, n=(self.image_shape ** 2) * self.image_dim)
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
        ind = ind.reshape((self.image_shape, self.image_shape, self.image_dim))

        image = (np.expand_dims(ind, 0))
        predictions = self.model.predict(image)[0]
        self.evaluations += 1

        if self.targeted:
            if np.argmax(predictions) == self.index:
                self.stop = True
                self.evaluation_found = self.evaluations

            target_prediction = predictions[self.index]
            predictions[self.index] = -999
            other_prediction_index = np.argmax(predictions)
            other_prediction = predictions[other_prediction_index]
        else:
            if np.argmax(predictions) != self.index:
                self.stop = True
                self.evaluation_found = self.evaluations

            other_prediction = predictions[self.index]
            predictions[self.index] = -999
            target_prediction_index = np.argmax(predictions)
            target_prediction = predictions[target_prediction_index]

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

    def attack(self, image, pop_size, targeted=True, index=0, num_eval=100000, draw=False):
        self.image = image
        self.index = index
        self.stop = False
        self.evaluations = 0
        self.evaluation_found = 0
        self.targeted = targeted

        # initialize population
        pop = self.toolbox.population(n=pop_size)
        flattened_image = image.flatten()
        images = [flattened_image] * pop_size

        for individual, image in zip(pop, images):
            for i in range(len(individual)):
                individual[i] += image[i]

        if draw:
            if not self.plot_img:
                plt.figure()
                plt.ion()
                plt.show()
                img = np.array(pop[0])
                if self.image_dim > 1:
                    img = img.reshape((self.image_shape, self.image_shape, self.image_dim))
                else:
                    img = img.reshape((self.image_shape, self.image_shape))
                self.plot_img = plt.imshow(img)
                plt.draw()
                plt.pause(0.001)

        while self.evaluations < num_eval:
            # evaluate the entire population
            fitnesses = map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            if self.stop:
                return self.evaluation_found

            # normalize for negative fitness values
            fits = [ind.fitness.values[0] for ind in pop]
            min_fit = abs(min(fits))
            for ind in pop:
                ind.fitness.values = (ind.fitness.values[0] + min_fit,)

            # get elite member
            elite = max(pop, key=lambda ind: ind.fitness.values)

            # select the next generation individuals
            offspring = [elite]
            maximum = sum(ind.fitness.values[0] for ind in pop)
            selection_probs = [ind.fitness.values[0] / maximum for ind in pop]
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
                ind = np.array(elite)
                if self.image_dim > 1:
                    ind = ind.reshape((self.image_shape, self.image_shape, self.image_dim))
                else:
                    ind = ind.reshape((self.image_shape, self.image_shape))
                self.plot_img.set_data(ind)
                plt.draw()
                plt.pause(0.00001)

        return num_eval
