Cross-Lingual Text Classification Tool for Multiple Classes
===========================================================

TOC
---

  * Requirements_
  * Installation_
  * Documentation_
     - HHTL_
  * References_

.. _Requirements:

Requirements
------------

To install clsa you need:

   * Python 2.7
   * Numpy (>= 1.11)
   * Sklearn (>= 0.17)

.. _Installation:

Installation
------------

To clone the repository run, 

   git clone git://github.com/thanhtunge6/cltc.git

.. _Documentation:

Documentation
-------------

.. _SHFR:

SHFR
~~~~

An implementation of Sparse Heterogeneous Feature Representation learning.
See [Joey2014]_ for a detailed description.

The data for cross-language sentiment classification that has been used in the above
study can be found here [#f1]_.

cltc_train
??????????

Training script for CLSA. See `./cltc_train --help` for further details. 

Usage::

    $ python ./cltc_train en de data/source.processed data/target.processed model.bz2

    |V_S| = 4535
    |V_T| = 3157
    classes = {0,1,2}
    Number of tasks = 6
    |s_train| = 600
    |t_train| = 300
    Train classifiers for source language
    Learn classifier  1
    Learn classifier  2
    Learn classifier  3
    Learn classifier  4
    Learn classifier  5
    Learn classifier  6
    Train source took  0.0757110118866
    Train classifiers for target language
    Learn classifier  1
    Learn classifier  2
    Learn classifier  3
    Learn classifier  4
    Learn classifier  5
    Learn classifier  6
    Train target took  0.0314400196075
    Mapping s_w and t_w
    Write model
    Writing model took  1.35773491859  sec




cltc_predict
????????????

Prediction script for CLTC.

Usage::

    $ python ./cltc_test model.bz2 data/target_test.processed


    Load model
    Loading model took  0.45977306366  sec
    |t_test| = 900
    Accuracy:  0.721111111111



.. _References:
References
----------

.. [#f1] http://www.uni-weimar.de/en/media/chairs/webis/corpora/corpus-webis-cls-10/

.. [Joey2014] Zhou, P. T., Pan, S. J., Tsang I. W. and Tan M. `Heterogeneous Domain Adaptation for Multiple Classes <http://www.jmlr.org/proceedings/papers/v33/zhou14.pdf>`_. In Proceedings of AISTATS 2014.
