import imp
import logging
import pandas as pd
import os

class GetCorpusKB:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def all_CSVs_to_df(self, extension = '.csv'): # Non default argument dump_dir must be before default arguments
        dfs = []
        try:
            for subdir, dirs, files in os.walk(self.input_dir): # os.walk() requires folder not file
                for filename in files:
                    ext = os.path.splitext(filename)[-1].lower()
                    if ext == extension:
                        input_file_dir = os.path.join(subdir, filename)
                        try:
                            df = pd.read_csv(input_file_dir, header = None)
                        except:
                            logging.error("Cannot read knowledgebase for language detection")

                        dfs.append(pd.DataFrame({"sample_text": df.iloc[:, 0], "intent": filename}))
            return pd.concat(dfs)
        except Exception as e:
            logging.error(e)