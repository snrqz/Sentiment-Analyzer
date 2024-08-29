import random as rnd
from analyzing_info import *
from preprocessing_txt import *
import numpy as np


class Data:
    def __init__(self, X, y, size: int):
        self.X = X
        self.y = y
        self.size = size

    def get_x(self): return self.X

    def get_size(self): return self.size

    def get_y(self): return self.y

    def set_x(self, X): self.X = X

    def set_y(self, y): self.y = y

    def set_size(self, size: int): self.size = size


class SplitData(Data):
    def __init__(self, X, y, size: int):
        super().__init__(X, y, size)

    def splitting_data(self):
        if self.size <= 0 or self.size > 1:
            raise ValueError(f"Cannot choose size: {self.size}, please re-insert")

        train_size = int(self.size * len(self.X))

        train_indices = set(rnd.sample(range(len(self.X)), train_size))
        all_indices = set(range(len(self.X)))
        test_indices = all_indices - train_indices

        X_train = self.X.iloc[list(train_indices)]
        y_train = self.y.iloc[list(train_indices)]
        X_test = self.X.iloc[list(test_indices)]
        y_test = self.y.iloc[list(test_indices)]

        return X_train, X_test, y_train, y_test


class BooleanModel:
    def __init__(self):
        self.hasFit = False
        self.booleanData = None

    def fit(self, X_train, y_train, text):
        if not self.hasFit:
            distances_data = label_distances(X_train, y_train, text)
            self.booleanData = convert_distances_to_booleans(distances_data, text)
            self.hasFit = True
        else:
            raise ValueError(
                "The model has already been fitted. Please reinitialize or create a new instance if you need to fit a different model.")

    def predict(self, X_test):
        if X_test is None or X_test.empty:
            raise ValueError("Insert a valid X_test param")

        if not self.hasFit or self.booleanData is None:
            raise ValueError("The model must be fitted before making predictions. Please fit the model first.")

        predicted_labels = []
        lemmatized_x_test = [lemmatize_text(x) for x in X_test]

        for x in lemmatized_x_test:
            dic_handler = {label: 0 for label in self.booleanData["_label_"]}
            for word in x.split():
                if word in self.booleanData:
                    for label, prob in zip(self.booleanData["_label_"], self.booleanData[word]):
                        dic_handler[label] += prob
            predicted_labels.append(max(dic_handler, key=dic_handler.get))

        return predicted_labels


class DistanceModel:
    def __init__(self):
        self.hasFit = False
        self.distancesData = None

    def fit(self, X_train, y_train, text):
        if not self.hasFit:
            self.distancesData = label_distances(X_train, y_train, text)
            self.hasFit = True
        else:
            raise ValueError(
                "The model has already been fitted. Please reinitialize or create a new instance if you need to fit a different model.")

    def predict(self, X_test):
        if not self.hasFit or self.distancesData is None:
            raise ValueError("The model must be fitted before making predictions. Please fit the model first.")

        if self.hasFit and not self.distancesData.empty:
            predicted_labels = []
            lemmatized_x_test = [lemmatize_text(x) for x in X_test]

            for x in lemmatized_x_test:
                dic_handler = {label: 0 for label in self.distancesData["_label_"]}
                for word in x.split():
                    if word in self.distancesData.columns:
                        for label, prob in zip(self.distancesData["_label_"], self.distancesData[word]):
                            dic_handler[label] += np.log(prob)

                predicted_labels.append(max(dic_handler, key=dic_handler.get))

            return predicted_labels

        raise ValueError("Model failed to predict")

