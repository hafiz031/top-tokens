import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import preprocessing
from get_corpus_n_knowledgebase import GetCorpusKB
from basic_preprocessing import BasicPreprocessor
import json
import os

class ClasswiseTFIDFVectorizer(TfidfTransformer):
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
    def __init__(self, input_dir, output_dir = "outputs"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.gckb = GetCorpusKB(self.input_dir)
        self.bp = BasicPreprocessor()
        self.le = preprocessing.LabelEncoder()

    # Create bag of words
    def tokenizer(self, text):
        return text.split()


    def transform_data(self):
        df = self.gckb.all_CSVs_to_df()
        df["sample_text"] = df["sample_text"].apply(lambda x: self.bp.do_basic_preprocessing(x))
        df["intent"] = self.le.fit_transform(df["intent"])
        df = df.dropna()
        docs_per_class = df.groupby(["intent"], as_index=False).agg({"sample_text": ' '.join})

        return docs_per_class


    def gen_ctfidf_matrix(self):
        docs_per_class = self.transform_data()
        count_vectorizer = CountVectorizer(tokenizer = self.tokenizer).fit(docs_per_class["sample_text"])
        count = count_vectorizer.transform(docs_per_class["sample_text"])
        self.words = count_vectorizer.get_feature_names()
        # print(self.words)
        ctfidf_vec = ClasswiseTFIDFVectorizer()
        ctfidf = ctfidf_vec.fit_transform(count, n_samples=len(docs_per_class)).toarray()

        return ctfidf

    
    def dump_intentwise_token_tfidf_priorities(self):
        ctfidf = self.gen_ctfidf_matrix()

        if not os.path.isdir(f"{self.output_dir}/intentwise_token_tfidf_priorities"):
            os.makedirs(f"{self.output_dir}/intentwise_token_tfidf_priorities")

        for actual_class_name in self.le.classes_:
            feature_index = ctfidf[self.le.transform([actual_class_name])[0]].nonzero()
            intentwise_priority = {}
            for i in feature_index[0]:
                # print(i, self.words[i], ctfidf[self.le.transform([actual_class_name])[0], i]) 
                intentwise_priority[self.words[i]] = ctfidf[self.le.transform([actual_class_name])[0], i]

                with open(f"{self.output_dir}/intentwise_token_tfidf_priorities/{''.join(actual_class_name.split('.')[:-1])}.json", "w") as jf:
                    json.dump(intentwise_priority, jf, ensure_ascii=False)


if __name__ == "__main__":
    ctfidf = ClasswiseTFIDF(input_dir = "intent-sample")
    ctfidf.dump_intentwise_token_tfidf_priorities()