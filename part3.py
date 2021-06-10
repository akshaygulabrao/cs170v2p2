import numpy as np
import pandas as pd


class Classifier:
    @staticmethod
    def train(f_name):
        global y
        with open(f_name, "r") as my_file:
            data = my_file.readlines()
        for i in range(len(data)):
            data[i] = data[i].split()
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = float(data[i][j])
                if j == 0:
                    data[i][j] = int(data[i][j])
        data = pd.DataFrame(data)

        # drop class label from x and create y
        x = data.drop(0, axis=1)
        y = data[0]

        # normalize data
        x = (x - x.mean() / x.std())
        return x, y

    def test(self, cols):
        global distance_matrix
        global subset_size
        subset_size = len(cols)
        x = self.x.loc[:, cols]
        x = np.array(x)
        distance_matrix = np.zeros((x.shape[0], x.shape[0]))
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[0]):
                distance_matrix[i][j] = np.linalg.norm(x[i] - x[j])
                if i == j:
                    distance_matrix[i][j] = np.Inf

    def __init__(self, f_name):
        self.x, self.y = self.train(f_name)