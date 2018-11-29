import random
import statistics
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from novelty_attack import NoveltyAttack
from zoo.setup_cifar import CIFAR, CIFARModel
from zoo.setup_inception import ImageNet, InceptionModel


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


# https://stackoverflow.com/a/34325723
# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    with tf.Session() as sess:
        use_log = True

        # does not work
        data, model = ImageNet(), InceptionModel(sess, use_log)

        # IDK image dims
        attack = NoveltyAttack(model, 32, 3, dist_delta=0.05)

        NUM_SAMPLES = 10
        TARGETED = True
        MAX_QUERIES = 1000000
        POP_SIZES = [6, 12, 36, 50, 100, 250, 500, 1000]
        K_VALUES = {6: [1, 2, 3],
                    12: [3, 6, 9],
                    36: [6, 12, 18],
                    50: [5, 15, 20],
                    100: [10, 20, 30],
                    250: [20, 45, 75],
                    500: [25, 50, 100],
                    1000: [25, 75, 100]}

        # POP_SIZES = [6]

        inputs, targets = generate_data(data, samples=NUM_SAMPLES, targeted=TARGETED,
                                        start=0, inception=False)

        inputs = inputs[1:NUM_SAMPLES + 1]
        targets = targets[1:NUM_SAMPLES + 1]

        filename = "results/nov_imagenet_run{0}.log"
        filename = filename.format("-targeted") if TARGETED else filename.format("")

        OUTPUT_FILE = open(filename, "w+")
        OUTPUT_FILE.truncate(0)

        pop_count = 0

        for pop_size in POP_SIZES:
            pop_count += 1

            for k in K_VALUES[pop_size]:
                queries = []
                times = []
                fails = 0

                print_progress_bar(0, len(inputs),
                                   prefix="pop {0} {1}/{2} k={3}".format(pop_size, pop_count, len(POP_SIZES), k),
                                   suffix="{0}/{1}".format(0, len(inputs)))

                for i in range(len(inputs)):
                    image = (np.expand_dims(inputs[i], 0))
                    prediction = model.predict(image)
                    original_index = np.argmax(prediction)
                    target_index = np.argmax(targets[i])

                    # print("sample", i + 1, "- changing", original_index, "to", target_index)

                    index = target_index if TARGETED else original_index

                    time_start = time.time()
                    query_count, adv = attack.attack(image=image, pop_size=6, k=k, targeted=TARGETED, index=index,
                                                     num_eval=MAX_QUERIES,
                                                     draw=False)
                    time_end = time.time()

                    # print("took", time_end - time_start, "seconds")
                    # print("")

                    if query_count != MAX_QUERIES:
                        queries.append(query_count)
                        times.append(time_end - time_start)
                    else:
                        fails += 1

                    print_progress_bar(i + 1, len(inputs),
                                       prefix="pop {0} {1}/{2} k={3}".format(pop_size, pop_count, len(POP_SIZES), k),
                                       suffix="{0}/{1}".format(i + 1, len(inputs)))

                OUTPUT_FILE.write("RESULTS\n")
                OUTPUT_FILE.write("--------------------------------------------------------\n")
                OUTPUT_FILE.write("population size={0}\n".format(pop_size))
                OUTPUT_FILE.write("k value={0}\n".format(k))
                OUTPUT_FILE.write("number of attacks={0}\n".format(len(inputs)))
                OUTPUT_FILE.write("failed attacks={0}\n".format(fails))
                OUTPUT_FILE.write("attack success rate={0}\n".format((len(inputs) - fails) / len(inputs) * 100))
                OUTPUT_FILE.write("mean query count={0}\n".format(statistics.mean(queries)))
                OUTPUT_FILE.write("median query count={0}\n".format(statistics.median(queries)))
                OUTPUT_FILE.write("mean runtime={0}\n".format(statistics.mean(times) / 3600))
                OUTPUT_FILE.write("total runtime={0}\n".format(sum(times) / 3600))
                OUTPUT_FILE.write("--------------------------------------------------------\n\n")
                OUTPUT_FILE.flush()

        OUTPUT_FILE.close()
