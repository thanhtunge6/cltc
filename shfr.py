__author__ = 'ngot0008'
import numpy as np
from sklearn.multiclass import OutputCodeClassifier
from sklearn import svm,linear_model


def generatecodebook(n, code_size):
    # generate the codebook base on the number of classes
    # n: number of classes
    # code_size: Percentage of the number of classes to be used to create the code book.
    clf = OutputCodeClassifier(svm.LinearSVC(random_state=0),code_size=code_size, random_state=0)
    X = np.array([np.arange(n)]).T
    y = np.arange(n)
    clf.fit(X, y)
    codebook = np.vstack({tuple(row) for row in clf.code_book_.T})
    newcodebook = []
    for row in codebook:
        if len(np.unique(row))>1:
            newcodebook.append(row)
    return np.array(newcodebook).T


def classify(codelabel,codebook):
    # Find actual class based on the output code
    similarity = []
    for i in range (0,len(codebook)):
        distance = codelabel - codebook[i]
        sim = len(codelabel)-len(np.nonzero(distance)[0])
        similarity.append(sim)
    label = np.argmax(similarity)
    return label


def trainclassifiers(data, codebook):
    # Output svm classifiers for every task
    classifiers = []
    for i in range(0, len(codebook[0])):
        print "Learn classifier ", i + 1
        label = codebook[:, i]
        clf = svm.LinearSVC()
        clf.fit(data, label)
        classifiers.append(clf)
    return classifiers


def mapping(w_source, w_target):
    # Learn the mapping between s_w and t_w
    # by minimizing (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    print "Mapping s_w and t_w"
    clf = linear_model.Lasso(alpha=0.1,positive=True,max_iter=2000)
    clf.fit(w_source, w_target)
    return clf