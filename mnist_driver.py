import random
import statistics
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from gen_attack import GenAttack
from zoo.setup_mnist import MNIST, MNISTModel


def show(img, name="output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = (img + 0.5) * 255
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            # print ('image label:', np.argmax(data.test_labels[start+i]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


with tf.Session() as sess:
    use_log = True

    data, model = MNIST(), MNISTModel("zoo/models/mnist", sess, use_log)

    attack = GenAttack(model)

    num_samples = 10

    inputs, targets = generate_data(data, samples=num_samples, targeted=True,
                                    start=0, inception=False)

    inputs = inputs[1:num_samples + 1]
    targets = targets[1:num_samples + 1]

    query_results = []
    time_results = []

    max_queries = 100000
    fails = 0

    for i in range(len(inputs)):
        image = (np.expand_dims(inputs[i], 0))
        prediction = model.predict(image)

        print("changing", np.argmax(prediction), "to", np.argmax(targets[i]))

        time_start = time.time()
        result = attack.attack(image=image, target_index=np.argmax(targets[i]), pop_size=6, num_eval=max_queries,
                               draw=False)
        time_end = time.time()

        print("Took", time_end - time_start, "seconds to run", len(inputs), "samples.")

        if result != max_queries:
            query_results.append(result)
            time_results.append(time_end - time_start)
        else:
            fails += 1

        print("num_attacks=", len(inputs))
        print("failed_attacks=", fails)
        print("asr=", (len(inputs) - fails) / len(inputs) * 100)
        print("median query count=", statistics.median(query_results))
        print("mean runtime=", statistics.mean(time_results) / 3600)
