from distutils.command.config import config
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from get_corpus_n_knowledgebase import GetCorpusKB
from basic_preprocessing import BasicPreprocessor
import pickle
import json
from config import SMOOTH_IDF
import os

class TFIDF:
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
        df = df.dropna(subset = ["sample_text"])
        return list(df["sample_text"])


    def gen_ctfidf_matrix(self):
        corpus = self.transform_data()
        vectorizer = TfidfVectorizer(tokenizer = self.tokenizer, smooth_idf = SMOOTH_IDF)
        X = vectorizer.fit_transform(corpus)

        return vectorizer, {"min": np.min(X), "max": np.max(X)}

    
    def dump_model_and_metadata(self):
        tfidf_vectorizer, global_min_max = self.gen_ctfidf_matrix()
        print(global_min_max)
        if not os.path.isdir(f"{self.output_dir}/tfidf_priorities"):
            os.makedirs(f"{self.output_dir}/tfidf_priorities")

        with open(f"{self.output_dir}/tfidf_priorities/vectorizer.pkl", 'wb') as fin:
            pickle.dump(tfidf_vectorizer, fin)

        with open(f"{self.output_dir}/tfidf_priorities/global_min_max.json", "w") as jf:
            json.dump(global_min_max, jf, ensure_ascii = False, indent = 4)
