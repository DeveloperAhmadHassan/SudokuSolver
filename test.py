# import the necessary packages
from Sudokunet import SudokuNet
# from keras.optimizers import Adam
# from keras.datasets import mnist
# import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a1", "--arg1", "---argument1", action="append", required=True, help="Argument 1")
ap.add_argument("-a2", action="append", required=True, help="Argument 2")
args = vars(ap.parse_args())

print("Hello World\n")
print(args)