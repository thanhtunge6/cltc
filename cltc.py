__author__ = 'ngot0008'

import numpy as np
import shfr
import optparse
from bow import vocabulary, disjoint_voc, load
from compress import compressed_dump, compressed_load
import time

class CLTCModel(object):
    def __init__(self, s_classifiers, t_classifiers, lasso_model):
        self.s_classifiers = s_classifiers
        self.t_classifiers = t_classifiers
        self.lasso_model = lasso_model
        self.s_voc = None
        self.t_voc = None
        self.codebook = None


class CLTCTrainer(object):
    def __init__(self, s_train, s_label, t_train, t_label):
        self.s_train = s_train
        self.s_label = s_label
        self.t_train = t_train
        self.t_label = t_label

    def train(self, alpha):
        print "Train classifiers for source language"
        start = time.time()
        s_classifiers = shfr.trainclassifiers(self.s_train, self.s_label)
        s_w = []
        for clf in s_classifiers:
            s_w.append(clf.coef_[0])
        print "Train source took ", time.time()-start
        print "Train classifiers for target language"
        start = time.time()
        t_classifiers = shfr.trainclassifiers(self.t_train, self.t_label)
        t_w = []
        for clf in t_classifiers:
            t_w.append(clf.coef_[0])
        print "Train target took ", time.time() - start
        lasso_model = shfr.mapping(np.array(s_w), np.array(t_w))
        return CLTCModel(s_classifiers=s_classifiers, t_classifiers=t_classifiers, lasso_model=lasso_model)


def train_args_parser():
    description = """Prefixes `s_` and `t_` refer to source and target language,\
     resp. Train and unlabeled files are expected to be in Bag-of-Words format.
        """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                         "s_lang t_lang s_train_file " \
                                         "s_unlabeled_file t_unlabeled_file " \
                                         "model_file",
                                   description=description)

    parser.add_option("-a",
                      dest="alpha",
                      help="regularization parameter alpha of lasso model",
                      default=0.1,
                      metavar="float",
                      type="float")


    parser.add_option("--max-unlabeled",
                      dest="max_unlabeled",
                      help="max number of unlabeled documents to read;" \
                           "-1 for unlimited.",
                      default=-1,
                      metavar="int",
                      type="int")

    return parser

def train():
    """Training script for CLSCL.
    TODO: different translators.
    """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 5:
        parser.error("incorrect number of arguments (use `--help` for help).")

    slang = argv[0]
    tlang = argv[1]

    fname_s_train = argv[2]
    fname_t_train = argv[3]

    max_unlabeled = 50000

    # Create vocabularies
    s_voc = vocabulary(fname_s_train, mindf=2, maxlines=max_unlabeled)
    t_voc = vocabulary(fname_t_train, mindf=2, maxlines=max_unlabeled)
    s_voc, t_voc, dim = disjoint_voc(s_voc, t_voc)
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    # print("|V| = %d" % dim)

    # Load labeled and unlabeled data
    s_train, s_labels, classes = load(fname_s_train, s_voc, dim)
    print("classes = {%s}" % ",".join(classes))
    print("|s_train| = %d" % s_train.shape[0])
    codebook = shfr.generatecodebook(len(classes), 5)
    print "Number of tasks = ", len(codebook)
    s_codelabels = []
    for i in range(0, len(s_labels)):
        label = int(s_labels[i])
        s_codelabels.append(codebook[label])
    s_codelabels = np.array(s_codelabels)

    t_train, t_labels, classes = load(fname_t_train, t_voc, dim)
    print("|t_train| = %d" % t_train.shape[0])
    t_codelabels = []
    for i in range(0, len(t_labels)):
        label = int(t_labels[i])
        t_codelabels.append(codebook[label])
    t_codelabels = np.array(t_codelabels)

    cltc_trainer = CLTCTrainer(s_train, s_codelabels, t_train, t_codelabels)

    model = cltc_trainer.train(options.alpha)
    model.s_voc = s_voc
    model.t_voc = t_voc
    model.dim = dim
    model.codebook = codebook
    compressed_dump(argv[4], model)


def predict_args_parser():
    """Create argument and option parser for the
    prediction script.
    """
    description = """Prefixes `s_` and `t_` refer to source and target language
    , resp. Train and unlabeled files are expected to be in Bag-of-Words format.
    """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "model_file " \
                                   "t_test_file",
                                   description=description)

    return parser


def predict():
    """Prediction script for CLSA.  """
    parser = predict_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 2:
        parser.error("incorrect number of arguments (use `--help` for help).")

    fname_model = argv[0]
    fname_t_test = argv[1]

    cltc_model = compressed_load(fname_model)

    s_classifiers = cltc_model.s_classifiers
    t_classifiers = cltc_model.t_classifiers
    lasso_model = cltc_model.lasso_model
    s_voc = cltc_model.s_voc
    t_voc = cltc_model.t_voc

    dim = cltc_model.dim
    codebook = cltc_model.codebook

    t_test, labels, classes = load(fname_t_test,t_voc)
    print("|t_test| = %d" % t_test.shape[0])
    predictions = []
    for i in range(0,len(s_classifiers)):
        t_classifiers[i].coef_=lasso_model.predict(s_classifiers[i].coef_)
        predictions.append(t_classifiers[i].predict(t_test))
    predictions = np.array(predictions).T
    y = []
    for code in predictions:
        y.append(shfr.classify(code,codebook))
    y == np.array(y)

    print "Accuracy: ",np.sum(y == labels)/float(len(y))

