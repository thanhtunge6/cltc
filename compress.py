import bz2
import gzip
import cPickle as pickle
import time


def compressed_dump(fname, model):
    """Pickle the model and write it to `fname`.
    If name ends with '.gz' or '.bz2' use the
    corresponding compressors else it pickles
    in binary format.
    Parameters
    ----------
    fname : str
        Where the model shall be written.
    model : object
        The object to be pickeled.
    """
    print "Write model"
    start = time.time()
    if fname.endswith(".gz"):
        f = gzip.open(fname, mode="wb")
    elif fname.endswith(".bz2"):
        f = bz2.BZ2File(fname, mode="w")
    else:
        f = open(fname, "wb")
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print "Writing model took ",time.time()-start," sec"


def compressed_load(fname):
    """Unpickle a model from `fname`. If `fname`
    endswith '.bz2' or '.gz' use the corresponding
    decompressor otherwise unpickle binary format.
    Parameters
    ----------
    fname : str
        From where the model shall be read.
    """
    print "Load model"
    start = time.time()
    if fname.endswith(".gz"):
        f = gzip.open(fname, mode="rb")
    elif fname.endswith(".bz2"):
        f = bz2.BZ2File(fname, mode="r")
    else:
        f = open(fname, "rb")
    model = pickle.load(f)
    f.close()
    print "Loading model took ", time.time() - start, " sec"
    return model