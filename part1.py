import numpy as np
from itertools import combinations


def forward_selection(features):
    num_features = len(features)
    current_choice = [([], np.random.rand())]
    max_accuracy = current_choice[-1][1]
    print("Using all features and a 'random' evaluation,\
  I get an accuracy of {0:.1%}".format(current_choice[-1][1]))
    print("\nBeginning search.\n")
    feature_added_list = []
    for j in range(num_features):
        # number of iterations of code that we must do
        for i in features:
            j = current_choice[-1][0]
            j.append(i)
            feature_added_list.append((list(j), np.random.rand()))
            j.pop()
        [print('Using feature(s) {0} accuracy is {1:.1%}'.format(i[0], i[1])) \
         for i in feature_added_list]

        c = lambda a: feature_added_list[a][1]
        index = max(range(len(feature_added_list)), key=c)
        a = feature_added_list[index]
        print("\nFeature set {0} was best, accuracy is {1:.1%}".format(a[0], a[1]))
        if a[1] < max_accuracy:
            print('Warning, Accuracy has decreased!\n')
            max_accuracy = a[1]
        else:
            print()
        current_choice.append(feature_added_list[index])
        feature_added_list.clear()
        features.pop(index)
    final = lambda f: current_choice[f][1]
    finalIndex = max(range(len(current_choice)), key=final)
    b = current_choice[finalIndex]
    print("Finished search!! The best feature subset is {0}, which has an accuracy of\
  {1:.1%}".format(b[0], b[1]))
    return current_choice[finalIndex][0]


def backward_elimination(features):
    num_features = len(features)
    current_choice = [(features, np.random.rand())]
    max_accuracy = current_choice[-1][1]
    print("Using all features and a \"random\" evaluation,\
  I get an accuracy of {0:.1%}".format(current_choice[-1][1]))
    print("\nBeginning Search")

    for j in reversed(range(4)):
        feature_removed_list = (list(combinations(features, j)))
        # sanitize clean up
        new_list = []
        for i in feature_removed_list:
            new_list.append((list(i), np.random.rand()))

        [print('Using feature(s) {0} accuracy is {1:.1%}'.format(i[0], i[1])) \
         for i in new_list]
        print()

        f = lambda i: new_list[i][1]
        index = max(range(len(new_list)), key=f)
        a = new_list[index]
        # set features, to the new_list that we chose
        features = a[0]
        print('Feature set {0} was best, accuracy is {1:.1%}'.format(a[0], a[1]))
        if a[1] < max_accuracy:
            print('Warning, Accuracy has decreased!\n')
            max_accuracy = a[1]
        else:
            print()
        current_choice.append(new_list[index])
        # print(current_choice)

    final = lambda i: current_choice[i][1]
    final_index = max(range(len(current_choice)), key=final)
    b = current_choice[final_index]
    print("Finished search!! The best feature subset is {0}, which has an accuracy of\
  {1:.1%}".format(b[0], b[1]))
    return current_choice[final_index][0]


print('Welcome to Akshay Gulabrao Feature Selection Algorithm')
print('\nPlease enter total # of features')
num_features = int(input())
# features = 4
features = list(np.arange(num_features))

print('\nType the number of the algorithm you want to run')
print('1. Forward Selection')
print('2. Backward Elimination')
choice = int(input())
# choice = 1

if choice == 1:
    forward_selection(features)
if choice == 2:
    backward_elimination(features)
