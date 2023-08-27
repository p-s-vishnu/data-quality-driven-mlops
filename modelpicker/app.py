import numpy as np
import sys
import os
from src.modelpicker import *

model_dict = {
    0: "Logistic Regression Classification",
    1: "k-Nearest Neighbors Classification",
    2: "Decision Tree Classification",
    3: "Multilayer Perceptron Classification",
    4: "Support Vector Machine Classification"
}

if __name__ == "__main__":

    # Read csv files
    FILENAME_PREDICTIONS = "data/polluted/telecomchurn/FeatureAccuracyPolluter/predictions0.1.csv"
    FILENAME_LABELSPACE = "data/polluted/telecomchurn/labelspace.csv"
    FILENAME_ORACLE = "data/polluted/telecomchurn/oracle.csv"

    print('-' * 40)
    print("filename_predictions:", FILENAME_PREDICTIONS)
    file_predictions = open(FILENAME_PREDICTIONS)
    predictions = np.loadtxt(file_predictions, delimiter=",")

    print("filename_labelspace:", FILENAME_LABELSPACE)
    file_labelspace = open(FILENAME_LABELSPACE)
    labelspace = np.loadtxt(file_labelspace, delimiter=",")

    print("load oracle file (true labels) for test mode:", FILENAME_ORACLE)
    file_oracle = open(FILENAME_ORACLE, 'r')
    oracle = np.loadtxt(file_oracle, delimiter=',')

    budget = int(len(predictions)*0.12)
    print("budget:", budget)

    (bestmodel, posterior_t) = modelpicker(predictions,
                labelspace,
                budget,
                oracle=oracle)
    print("Best model ID: " + str(bestmodel))
    print("Best model:", model_dict[bestmodel])