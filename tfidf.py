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
from config import CALCULATE_FOR_EACH_CLASS_SEPARATELY
import os

class TFIDF:
    def __init__(self, input_dir, output_dir = "outputs"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.gckb = GetCorpusKB(self.input_dir)
        self.bp = BasicPreprocessor()
        #self.le = preprocessing.LabelEncoder()


    # Create bag of words
    def tokenizer(self, text):
        return text.split()


    def transform_data(self):
        df = self.gckb.all_CSVs_to_df()
        # WARNING: We didn't remove stopwords here, as for final labelling no stopwords will be removed, if 
        # we removed it here, therefore, during labelling some of the tfidf values may exceed 1 (after min-max normalization)
        # and the RGBColor() would through error (as it would get color value more than 255)
        # TODO: Need to find a better solution for above case.
        df["sample_text"] = df["sample_text"].apply(lambda x: self.bp.do_basic_preprocessing(x, remove_predefined_stopwords = False))

        if CALCULATE_FOR_EACH_CLASS_SEPARATELY:
            df = df.dropna()
        else:
            df = df.dropna(subset = ["sample_text"])
        
        if CALCULATE_FOR_EACH_CLASS_SEPARATELY:
            docs_per_class = {}
            intents = set(df["intent"])
            for intent in intents:
                docs_per_class[intent] = list(df[df["intent"] == intent]["sample_text"])
            return docs_per_class
        else:
            return {"combined": list(df["sample_text"])}


    def gen_tfidf_matrix(self):
        corpus = self.transform_data()

        if CALCULATE_FOR_EACH_CLASS_SEPARATELY:
            vectorizers = {}
            min_maxes = {}
            for intent in corpus.keys():
                # If we put TfidfVectorizer outside of the loop, it seems to create problems (surprisingly), 
                # and the tfidf values won't match if we transform the exact same examples
                # inside generate_annotated_output()
                vectorizer = TfidfVectorizer(tokenizer = self.tokenizer, smooth_idf = SMOOTH_IDF)
                X = vectorizer.fit_transform(corpus[intent])
                vectorizers[intent] = vectorizer
                min_maxes[intent] = {"min": np.min(X), "max": np.max(X)}
            return vectorizers, min_maxes
        else:
            X = vectorizer.fit_transform(corpus["combined"])
            return {"combined": vectorizer}, {"combined": {"min": np.min(X), "max": np.max(X)}}

    
    def dump_model_and_metadata(self):
        tfidf_vectorizer, global_min_max = self.gen_tfidf_matrix()

        if not os.path.isdir(f"{self.output_dir}/tfidf_priorities"):
            os.makedirs(f"{self.output_dir}/tfidf_priorities")

        with open(f"{self.output_dir}/tfidf_priorities/vectorizer.pkl", 'wb') as fin:
            pickle.dump(tfidf_vectorizer, fin)

        with open(f"{self.output_dir}/tfidf_priorities/global_min_max.json", "w") as jf:
            json.dump(global_min_max, jf, ensure_ascii = False, indent = 4)
