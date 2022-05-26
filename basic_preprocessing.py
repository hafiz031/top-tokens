from unicodedata import category
import re
import pandas as pd
import os
from config import *
import logging


predefined_stopwords = None

def load_stopwords(force_loading = False):
    global predefined_stopwords
    if REMOVE_PREDEFINED_STOPWORDS_DURING_GENERATING_TOP_TOKENS or force_loading:
        if os.path.exists('stopwords.csv'):
            try:
                stopword_df = pd.read_csv('stopwords.csv', header = None) # without header = None the first row will be removed assuming it was the header (although it is not, it has no header)
                predefined_stopwords = set(' '.join(list(stopword_df.iloc[:, 0])).split())
            except Exception as e:
                logging.warning("An error occurred while loading predefined stopword-knowledgebase. If you want the training and prediction to make use of stopwords, please "
                        "consider putting a file: 'stopwords.csv' where the file contains stopwords.")
                predefined_stopwords = None
            else:
                logging.info("Predefined stopword-knowledgebase found and loaded.")
        else:
            logging.warning("Predefined stopword-knowledgebase is not found. If you want the training and prediction to make use of stopwords, please "
                        "consider putting a file: 'stopwords.csv' where the file contains stopwords.")

class BasicPreprocessor:
    def __init__(self):
        load_stopwords()

    def to_lowercase(self, text):
        return text.lower()

    def remove_punctuations(self, text):
        return ''.join(ch if not category(ch).startswith('P') \
                or ch == '_' or ch == '-' or ch == "'" or ch == "’" \
                else ' ' for ch in text)

    def remove_numbers(self, text):
        return re.sub('[0-9০-৯]+', '', text)
    

    def remove_newlines(self, text):
        return text.replace('\n', ' ')

    
    def remove_extra_whitespaces(self, text):
        return ' '.join(text.split())
    
    
    def remove_remainder(self, text, cons = 2): 
        """"Remove remainder consecutive character, if consecutive character count is > cons"""
        text = str(text) 
        output = '' 
        prev_ch = '' 
        cons_count = 1 
        for ch in text:
            if re.search(r'[0-9০-৯]', ch) is not None:
                prev_ch = '' # We don't want to remove repetition in number
            if prev_ch != ch: 
                cons_count = 1 
            else: 
                cons_count += 1 
            if cons_count <= cons: 
                output += ch 
            prev_ch = ch 
        return output 


    def do_basic_preprocessing(self, text, remove_punc = True, remove_num = True, remove_predefined_stopwords = True, lowercasing = True):
        if lowercasing:
            text = self.to_lowercase(text)
        text = self.remove_newlines(text)

        if remove_punc:
            text = self.remove_punctuations(text)

        if remove_num:
            text = self.remove_numbers(text)

        if remove_predefined_stopwords and predefined_stopwords is not None:
            if os.path.exists('stopwords.csv'):
                l_word = text.split()
                text = ' '.join([word for word in l_word if word not in predefined_stopwords])

        text = self.remove_remainder(text = text)

        text = self.remove_extra_whitespaces(text)
        return text

