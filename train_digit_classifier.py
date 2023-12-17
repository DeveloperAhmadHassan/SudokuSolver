# import the necessary packages
import numpy as np
from keras import metrics
from keras.src import losses
import os

from Sudokunet import SudokuNet
from keras.optimizers import Adam
from keras.datasets import mnist
# import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

from constants import INIT_LR, BS, EPOCHS, WIDTH, HEIGHT, DEPTH, CLASSES, VERBOSE


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model after training")
args = vars(ap.parse_args())

print("[INFO] accessing MNIST...")
((x_train, y_train), (x_test, y_test)) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# TODO: See metrics attributes
metrics = [
    metrics.Accuracy(name="Accuracy"),
    metrics.AUC(name="Area_Under_Curve"),
    metrics.MeanSquaredError(name="Mean_Squared_Error"),
    # metrics.F1Score(name="F1_Score")
]

# TODO: See optimizer attributes
optimizer = Adam(
    name="Adam",
    learning_rate=INIT_LR
)

# TODO: See loss attributes
loss = losses.MeanSquaredError(
    name="Categorical_Cross_Entropy"
)

model = SudokuNet.build(width=WIDTH, height=HEIGHT, depth=DEPTH, classes=CLASSES)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

H = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    validation_split=3,
    batch_size=BS,
    epochs=EPOCHS,
    verbose=VERBOSE)

print(H)

# evaluate the network

# evaluate = model.evaluate(x_test, y_test, verbose=VERBOSE, use_multiprocessing=True)
# print(evaluate)
# print(model.metrics_names)

predictions = model.predict(x_test)
# INFO: https://stackoverflow.com/a/54595455
# TODO:
# ERROR:
# CODE_EXP:

#
print(classification_report(
    np.argmax(y_test, axis=1),
    np.argmax(predictions, axis=1),
    target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

model.save(
    filepath=args["model"],
    save_format="h5",
    overwrite=True
)
