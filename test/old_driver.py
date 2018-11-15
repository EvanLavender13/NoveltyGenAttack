import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from deap import base
from deap import creator
from deap import tools
from tensorflow import keras

### TensorFlow stuff
from NoveltyGenAttack.test import operators

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

image = train_images[0]

IMG_SIZE = len(image)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print(test_labels[0])

# operators.bernoulli_2d(image, 0.5, 1.0 * 0.3)

DELTA = 1.0 * 0.1

np.random.seed(64)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Individual is a list
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# uniform sample
toolbox.register("uniform_sample", np.random.uniform, low=-DELTA, high=DELTA)

# create a random sample
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.uniform_sample, n=IMG_SIZE ** 2)

# create a population of random samples
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval(individual):
    # Do some hard computing on the individual
    return (sum(individual),)


# operators
toolbox.register("evaluate", eval)

images = []
pop = toolbox.population(n=25)
for ind in pop:
    image_copy = np.ndarray.copy(image)
    operators.apply_noise(image_copy, ind)
    images.append(image_copy)

plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
plt.show(1)

# Evaluate the entire population
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

CXPB, MUTPB, NGEN = 0.5, 0.2, 100
for g in range(NGEN):
    image_copy = np.ndarray.copy(image)
    operators.apply_noise(image_copy, pop[0])
    img = (np.expand_dims(image_copy, 0))

    predictions = model.predict(img)
    print(np.argmax(predictions))
    print(test_labels[0])
    # Select the next generation individuals
    offspring = tools.selRoulette(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.random() < CXPB:
            fitness_child1 = child1.fitness.values[0]
            fitness_child2 = child2.fitness.values[0]
            tools.cxUniform(child1, child2, fitness_child1 / (fitness_child1 + fitness_child2))
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.random() < MUTPB:
            operators.mutate(mutant, 0.5, DELTA)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    # print("  Min %s" % min(fits))
    # print("  Max %s" % max(fits))
    print("gen: " + str(g) + "  Avg %s" % mean)
    # print("  Std %s" % std)
    # print()

images = []
for ind in pop:
    image_copy = np.ndarray.copy(image)
    operators.apply_noise(image_copy, ind)
    images.append(image_copy)

plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
plt.show(1)
