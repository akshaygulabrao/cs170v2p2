import numpy as np
import time
import pandas as pd

# parse text file into pandas dataframe

f = ['cs_170_small80.txt', 'cs_170_large80.txt']
global distance_matrix
global subset_size
global y
distance_matrix = 0


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


# makes sense to only have a validator class given my specific
# implementation
class Validator:

    def leave_one_out_validation(self):
        global subset_size
        if not subset_size:
            return len(np.unique(self.y)) ** -1
        # test the leave 1 out
        total = 0
        correct = 0
        for i in range(y.shape[0]):
            min_dist = np.argmin(self.dist[i])
            if y[min_dist] == y[i]:
                correct += 1
            total += 1
        return correct / total

    def __init__(self):
        global distance_matrix
        global y
        self.dist = distance_matrix
        self.y = y


print("Using given feature subset on small dataset:")
start = time.time()
c = Classifier(f[0])
end = time.time()
print("Parsing Data took", end - start, "seconds")
start = time.time()
c.test([3, 5, 7])
end = time.time()
print("Creating distance matrix took", end - start, "seconds")
start = time.time()
v = Validator()
print(v.leave_one_out_validation())
end = time.time()
print("Leave-1-out validation took", end - start, "seconds")

print("\nUsing given feature subset on large dataset")
start = time.time()
c = Classifier(f[1])
end = time.time()
print("Parsing data took", end - start, "seconds")
start = time.time()
c.test([1, 15, 27])
end = time.time()
print("Creating distance matrix took", end - start, "seconds")
start = time.time()
v = Validator()
print(v.leave_one_out_validation())
end = time.time()
print("Leave-1-out validation took", end - start, "seconds")
