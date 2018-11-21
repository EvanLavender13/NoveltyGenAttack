import random
import statistics
from timeit import default_timer as timer

import tensorflow as tf

from gen_attack import GenAttack

mnist = tf.keras.datasets.mnist

mnist_model = tf.keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=1)
#model.evaluate(x_test, y_test)

gen = GenAttack(model)

query_results = []
time_results = []

num_attacks = 1
max_queries = 10000
failed_attacks = 0.0

print(type(x_test))
print(type(x_test[0]))

for i in range(num_attacks):
    target = random.randint(0, 9)
    while target == y_test[i]:
        target = random.randint(0, 9)

    print("changing", y_test[i], "to", target)
    start = timer()
    query_result = gen.attack(image=x_test[i], index=y_test[i], target_index=target, pop_size=6, num_eval=max_queries,
                              draw=False)
    stop = timer()

    if query_result != max_queries:
        query_results.append(query_result)
        time_results.append(stop - start)
    else:
        failed_attacks += 1

print("num_attacks=", num_attacks)
print("failed_attacks=", failed_attacks)
print("asr=", (num_attacks - failed_attacks) / num_attacks * 100)
print("median query count=", statistics.median(query_results))
print("mean runtime=", statistics.mean(time_results) / 3600)
