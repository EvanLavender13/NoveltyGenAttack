import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools

from novelty_eval import NoveltyEvaluator


class NoveltyAttack:
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
        self.pop = None
        self.predictions = None
        self.targeted = True

        self.plot_img = None

        self.novelty_eval = None
        self.toolbox = base.Toolbox()

        np.random.seed(64)

        pool = multiprocessing.Pool()
        #self.toolbox.register("map", pool.map)

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

        behavior = self.predictions.get(tuple(individual))

        novelty = self.novelty_eval.evaluate_novelty(self.pop, self.predictions, behavior)

        if self.targeted:
            target_prediction = behavior[self.index]
            # print("novelty=", novelty, "target=", target_prediction)

            # --- minimal criteria ---
            # encourage increases to target_prediction
            novelty += (target_prediction * 1000000)

        # print("novelty=", novelty)

        return (novelty,)

    def distribution(self):
        """
        A bernoulli distribution in the given range of distortion

        Returns:
            Bernoulli(p) * U(-dist, dist)
        """
        return np.random.binomial(n=1, p=self.mut_prob) * np.random.uniform(low=-self.dist_delta,
                                                                            high=self.dist_delta)

    def attack(self, image, pop_size, k=30, targeted=True, index=0, num_eval=100000, draw=False):
        self.image = image
        self.index = index
        self.stop = False
        self.evaluations = 0
        self.evaluation_found = 0
        self.targeted = targeted

        self.novelty_eval = NoveltyEvaluator(pop_size, k)

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
                img = np.array(image)
                if self.image_dim > 1:
                    img = img.reshape((self.image_shape, self.image_shape, self.image_dim))
                else:
                    img = img.reshape((self.image_shape, self.image_shape))
                self.plot_img = plt.imshow(img)
                plt.draw()
                plt.pause(0.001)

        while self.evaluations < num_eval:
            self.pop = pop
            self.predictions = {}
            for i in range(len(pop)):
                individual = pop[i]
                ind = np.array(individual)
                ind = ind.reshape((self.image_shape, self.image_shape, self.image_dim))

                img = (np.expand_dims(ind, 0))
                prediction = self.model.predict(img)
                self.predictions[tuple(individual)] = prediction[0]
                self.evaluations += 1

                # print("pred=", prediction[0])
                # print("max=", np.max(prediction[0]))
                # print("target=", prediction[0][self.target_index])
                if self.targeted:
                    if np.argmax(prediction) == self.index:
                        self.stop = True
                        self.evaluation_found = self.evaluations
                        self.adversarial_image = individual
                else:
                    if np.argmax(prediction) != self.index:
                        self.stop = True
                        self.evaluation_found = self.evaluations
                        self.adversarial_image = individual

            # evaluate the entire population
            fitnesses = map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            self.novelty_eval.post_evaluation()

            if self.stop:
                return self.evaluation_found, self.adversarial_image

            # get elite member
            elite = max(pop, key=lambda ind: ind.fitness.values[0])

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

        return num_eval, None
