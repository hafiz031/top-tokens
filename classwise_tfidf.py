import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import preprocessing


class ClasswiseTFIDFVectorizer:
    def __init__(self, *args, **kwargs):
        super(ClasswiseTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) """
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape = (n_features, n_features),
                                  format = 'csr',
                                  dtype = np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag
        X = normalize(X, norm='l1', copy=False)
        return X


class ClasswiseTFIDF:
    def __init__(self):
        pass


    