import random

from gen_attack import GenAttack
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

mnist_model = tf.keras

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)

gen = GenAttack(model)

results = []

for i in range(10):
    target = random.randint(0, 9)
    while target == y_test[i]:
        target = random.randint(0, 9)

    print("changing", y_test[i], "to", target)
    result = gen.attack(image=x_test[i], index=y_test[i], target_index=3, pop_size=6, num_eval=100000, draw=False)
    results.append(result)
    print("finished after", result, "evaluations")

print(results)
print("mean=", sum(results) / len(results))
