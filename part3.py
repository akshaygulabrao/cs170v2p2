import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                if i == j: distance_matrix[i][j] = np.Inf
                elif distance_matrix[j][i] != 0:
                    distance_matrix[i][j] = distance_matrix[j][i]
                else: distance_matrix[i][j] = np.linalg.norm(x[i] - x[j])
        self.dist =  distance_matrix

    def forward_selection(self):
        # we start with the empty list
        c.test([])
        v = Validator(c.dist,[],self.y)
        greedy_choice_list = [tuple(([], v.acc))]
        # list of features to choose from
        feature_list = np.arange(1, self.x.shape[1] + 1)
        # while we haven't tried incorporating every features
        while feature_list.shape[0]:
            # generate prospects to add to our feature choice list
            prospects = [greedy_choice_list[-1][0] + [i] for i in feature_list]
            for i in range(len(prospects)):
                c.test(prospects[i])
                v = Validator(c.dist, prospects[i], self.y)
                prospects[i] = tuple((prospects[i], v.acc))
                print(prospects[i])
            greedy = lambda best : prospects[best][1]
            greedy_index = max(range(len(prospects)), key = greedy)
            feature_list = np.delete(feature_list, greedy_index)
            print("\nBest Choice: ", prospects[greedy_index])2
            greedy_choice_list.append(prospects[greedy_index])
        final = lambda best : greedy_choice_list[best][1]
        final_index = max(range(len(greedy_choice_list)),key = final)
        return greedy_choice_list[final_index]

    def backward_elimination(self):
        feature_list = np.arange(1,self.x.shape[1] + 1)
        c.test(feature_list)
        v = Validator(c.dist, feature_list, self.y)
        greedy_choice_list = [tuple((feature_list, v.acc))]
        while len(feature_list) > 0:
            prospects = [np.delete(greedy_choice_list[-1][0][:],i) for i in range(len(feature_list))]
            for i in range(len(prospects)):
                c.test(prospects[i])
                v = Validator(c.dist, prospects[i], self.y)
                prospects[i] = tuple((prospects[i], v.acc))
                print(prospects[i])
            if len(prospects):
                greedy = lambda best: prospects[best][1]
                greedy_index = max(range(len(prospects)), key=greedy)
                feature_list = prospects[greedy_index][0]
                print("\nBest Choice: ", prospects[greedy_index])
                greedy_choice_list.append(prospects[greedy_index])
        c.test([])
        v = Validator(c.dist, [], self.y)
        greedy_choice_list.append(([], v.acc))
        final = lambda best: greedy_choice_list[best][1]
        final_index = max(range(len(greedy_choice_list)), key=final)
        return greedy_choice_list[final_index]

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

c = Classifier(f[2])
# class1 =[]
# class2 = []
# for i in range(len(c.y)):
#     if c.y[i] == 2:
#         class2.append((c.x[16][i],c.x[37][i]))
#     else:
#         class1.append((c.x[16][i],c.x[37][i]))
# c1x = [i[0] for i in class1]
# c1y = [i[1] for i in class1]
# c2x = [i[0] for i in class2]
# c2y = [i[1] for i in class2]
# plt.scatter(c1x,c1y,color = "r",label="class1")
# plt.scatter(c2x,c2y,color = "blue",label = "class2")
# plt.xlabel("Feature 16")
# plt.ylabel("Feature 37")
# plt.legend()
# plt.title("Large Personal Dataset")
# plt.show()
# c = Classifier(f[3])
print("Welcome to Akshay Gulabrao's Feature Selection Algorithm")
print(f)
index = int(input("type the index of the file you want to test"))
c = Classifier(f[index])
alg = int(input("1 : Forward Selection\n2 : Backward Elimination"))
if alg == 1:
    print(c.forward_selection())
else:
    print(c.backward_elimination())
# print(c.backward_elimination())


