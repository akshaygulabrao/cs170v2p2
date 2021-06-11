import numpy as np
import pandas as pd

f = ['cs_170_small80.txt', 'cs_170_large80.txt','cs_170_small25.txt', 'cs_170_large25.txt']
class Classifier:
    @staticmethod
    def train(f_name):
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
        x = self.x.loc[:, cols]
        x = np.array(x)
        distance_matrix = np.zeros((x.shape[0], x.shape[0]))
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[0]):
                distance_matrix[i][j] = np.linalg.norm(x[i] - x[j])
                if i == j:
                    distance_matrix[i][j] = np.Inf
        self.dist =  distance_matrix

    def forward_selection(self):
        greedy_choice_list = [[]]
        feature_list = np.arange(1, self.x.shape[1] + 1)
        prospects = [greedy_choice_list[-1] + [i] for i in feature_list]
        for i in range(len(prospects)):
            c.test(prospects[i])
            v = Validator(c.dist,prospects[i], self.y)
            prospects[i] = tuple((prospects[i], v.acc))

        c.test([3, 5, 7]); v = Validator(c.dist,[3, 5, 7],self.y); print(v.acc)
        return 0

    def backward_elimination(self):
        return 0

    def __init__(self, f_name):
        self.x, self.y = self.train(f_name)


class Validator:
    def __init__(self, dist, cols, y):
        if len(cols) == 0:
            self.acc = len(np.unique(y)) ** -1
            return
        total = 0
        correct = 0
        for i in range(y.shape[0]):
            min_dist = np.argmin(dist[i])
            if y[min_dist] == y[i]:
                correct += 1
            total += 1
        self.acc = correct / total


c = Classifier(f[0])
c.forward_selection()

