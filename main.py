import numpy as np
import time
import pandas as pd

# parse text file into pandas dataframe

f = ['cs_170_small80.txt', 'cs_170_large_80.txt']
global distance_matrix
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
        if self.dist.shape[1] == 0:
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


c = Classifier(f[0])
c.test([1, 3, 5])
v = Validator()
print(v.leave_one_out_validation())



"""
xc = x.copy()
#forward selection 
# start current_choice list
def forward_selection(x):
    current_choice=[([],test(x,[]))]
    max_accuracy = current_choice[-1][1]
    xcols = list(x.columns)
    while len(xcols):
    # main loop
        feature_added_list = []
        for i in xcols:
            j = current_choice[-1][0]
            j.append(i)
            tmp = (list(j),test(x,list(j)),i)
            print(tmp[0],tmp[1])
            feature_added_list.append(tmp)
            j.pop()
        c = lambda a: feature_added_list[a][1]
        index = max(range(len(feature_added_list)),key = c)
        a = feature_added_list[index]
        current_choice.append(a)
        feature_added_list.clear()
        xcols.remove(a[2])
    final = lambda f : current_choice[f][1]
    finalIndex = max(range(len(current_choice)),key = final)
    b = current_choice[finalIndex]
    return b[0],b[1]
fname = "cs_170_small80.txt"
x,y = parse_test(fname)
print('Accuracy of small dataset using all features, time(seconds)')
s = time.time()
x = validator(x,y,np.arange(1,x.shape[1] + 1))
e = time.time()
print(x,',',e-s)
fname = "cs_170_large80.txt"
x,y = parse_test(fname)

print('Accuracy of large dataset using all features , time(seconds)')
s = time.time()
x = validator(x,y,np.arange(1,x.shape[1] + 1))
e = time.time()
print(x,',',e-s)
"""
