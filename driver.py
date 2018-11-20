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

gen.attack(orig_img=x_train[0], orig_index=y_train[0], target_index=3, pop_size=6, num_gen=1000)
