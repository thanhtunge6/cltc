import shfr, numpy as np
from bow import vocabulary,disjoint_voc,load
s_voc = vocabulary("data/source.processed", mindf=2, maxlines=1000)
t_voc = vocabulary("data/target.processed", mindf=2, maxlines=1000)
s_voc, t_voc, dim = disjoint_voc(s_voc, t_voc)
print("|V_S| = %d" % len(s_voc))
# Load labeled and unlabeled data
s_train, s_labels, classes = load("data/source.processed", s_voc)
t_train, t_labels, classes = load("data/target.processed", t_voc)
codebook = shfr.generatecodebook(len(classes),5)
s_codelabels = []
for i in range(0,len(s_labels)):
    label = int(s_labels[i])
    s_codelabels.append(codebook[label])
s_codelabels = np.array(s_codelabels)

print("|s_test| = %d" % t_train.shape[0])
t_codelabels = []
for i in range(0, len(t_labels)):
    label = int(t_labels[i])
    t_codelabels.append(codebook[label])
t_codelabels = np.array(t_codelabels)

s_classifiers = shfr.trainclassifiers(s_train, s_codelabels)
s_w = []
for clf in s_classifiers:
    s_w.append(clf.coef_[0])
print "Train classifiers for target language"
t_classifiers = shfr.trainclassifiers(t_train, t_codelabels)
t_w = []
for clf in t_classifiers:
    t_w.append(clf.coef_[0])
print np.array(s_w).shape
print "Learn mapping"
lasso_model = shfr.mapping(np.array(s_w), np.array(t_w))

t_test, testlabels, classes = load("data/target_test.processed", t_voc)
print "Predicting"
predictions = []
for i in range(0, len(s_classifiers)):
    t_classifiers[i].coef_ = lasso_model.predict(s_classifiers[i].coef_)
    predictions.append(t_classifiers[i].predict(t_test))
predictions = np.array(predictions).T
y = []
for code in predictions:
    y.append(shfr.classify(code, codebook))
y == np.array(y)

print "Accuracy: ", np.sum(y == testlabels) / float(len(y))

