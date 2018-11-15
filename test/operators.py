import numpy as np

def bernoulli_2d(image, p, delta):
    for x in range(len(image)):
        for y in range(len(image)):
            value = np.random.binomial(n=1, p=p) * np.random.uniform(low=-delta, high=delta)
            image[x][y] += value

    return image

def mutate(individual, p, delta):
    SIZE = len(individual)
    for x in range(SIZE):
        value = np.random.binomial(n=1, p=p) * np.random.uniform(low=-delta, high=delta)
        individual[x] += value

def apply_noise(image, noise):
    IMG_SIZE = len(image)
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            image[x][y] += noise[(x * IMG_SIZE) + y]
