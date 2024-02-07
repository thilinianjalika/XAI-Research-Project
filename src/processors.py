import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize, toktok
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from typing import List, Union
import joblib
from scipy.sparse._csr import csr_matrix


class LUTLabelEncoder:
    def __init__(self, labels: List[str]) -> None:
        self.lut = labels

    def transform(self, df: pd.DataFrame) -> np.array:
        enc_lbls = df.apply(lambda st: self.lut.index(st)).to_numpy()
        return enc_lbls

    def inverse_tranform(self, labels: List[int]) -> List[str]:
        labels = [self.lut[lbl] for lbl in labels]
        return labels


class Preprocessor:
    def _strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        return text

    def _remove_special_characters(self, text, remove_digits=True):
        pattern = r"[^a-zA-z0-9\s]"
        text = re.sub(pattern, "", text)
        return text

    def _remove_stopwords(self, text, is_lower_case=False):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [
                token for token in tokens if token not in self.stop_words
            ]
        else:
            filtered_tokens = [
                token for token in tokens if token.lower() not in self.stop_words
            ]
        filtered_text = " ".join(filtered_tokens)
        return filtered_text

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith("J"):
            return "a"  # Adjective
        elif treebank_tag.startswith("V"):
            return "v"  # Verb
        elif treebank_tag.startswith("N"):
            return "n"  # Noun
        elif treebank_tag.startswith("R"):
            return "r"  # Adverb
        else:
            return "n"  # Default to noun

    def _lemmatize_text(self, text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)  # Perform POS tagging
        lemmatized_words = [
            self.lemmatizer.lemmatize(word, pos=self._get_wordnet_pos(pos_tag))
            for word, pos_tag in pos_tags
        ]  # Lemmatize words with their respective POS tags
        lemmatized_text = " ".join(lemmatized_words)
        # cleaned_text = re.sub(
        #     r"\s*([.,!?:;])", r"\1", lemmatized_text
        # )  # Apply regex to clean the text ("Hello world !" -> "Hello world!")

        return lemmatized_text

    def __init__(self):
        # download the corpus
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("tagsets", quiet=True)
        except:
            pass

        # initialize
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = toktok.ToktokTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, txt: str) -> str:
        processed_txt = txt.lower()
        processed_txt = self._strip_html(processed_txt)
        processed_txt = self._remove_special_characters(processed_txt)
        processed_txt = self._remove_stopwords(processed_txt)
        processed_txt = self._lemmatize_text(processed_txt)
        return processed_txt


class TextVectorizer:
    def __init__(self, tfidf_path: str) -> None:
        self.preproc = Preprocessor()
        self.vectorizer = joblib.load(tfidf_path)

    def __call__(self, txts: Union[List[str], str]) -> csr_matrix:
        if type(txts) == str:
            txts = [txts]

        preprocs = [self.preproc(txt) for txt in txts]  # l sentences
        vects = self.vectorizer.transform(preprocs)  # matrix of shape (l, n)

        return vects
