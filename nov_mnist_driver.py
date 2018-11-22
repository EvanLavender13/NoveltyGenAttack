import random
import statistics
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from gen_attack import GenAttack
from novelty_attack import NoveltyAttack
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


if __name__ == "__main__":
    with tf.Session() as sess:
        use_log = True

        data, model = MNIST(), MNISTModel("models/mnist", sess, use_log)

        attack = NoveltyAttack(model, 28, 1)

        num_samples = 11
        targeted = False

        inputs, targets = generate_data(data, samples=num_samples, targeted=targeted,
                                        start=0, inception=False)

        inputs = inputs[1:num_samples + 1]
        targets = targets[1:num_samples + 1]

        query_results = []
        time_results = []

        max_queries = 100000
        fails = 0

        print("running", num_samples, "samples")

        for i in range(len(inputs)):
            image = (np.expand_dims(inputs[i], 0))
            prediction = model.predict(image)
            original_index = np.argmax(prediction)
            target_index = np.argmax(targets[i])

            print("sample", i + 1, "- changing", np.argmax(prediction), "to", np.argmax(targets[i]))

            index = target_index if targeted else original_index

            time_start = time.time()
            result = attack.attack(image=image, pop_size=20, targeted=targeted, index=index, num_eval=max_queries,
                                   draw=True)
            time_end = time.time()

            print("took", time_end - time_start, "seconds")
            print("")

            if result != max_queries:
                query_results.append(result)
                time_results.append(time_end - time_start)
            else:
                fails += 1

        print("")
        print("RESULTS")
        print("--------------------------------------------------------")
        print("num_attacks=", len(inputs))
        print("failed_attacks=", fails)
        print("asr=", (len(inputs) - fails) / len(inputs) * 100)
        print("mean query count=", statistics.mean(query_results))
        print("median query count=", statistics.median(query_results))
        print("mean runtime=", statistics.mean(time_results) / 3600)
        print("--------------------------------------------------------")
