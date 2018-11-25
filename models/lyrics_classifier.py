import _pickle as cPickle
import logging
import os
import os.path

import nltk
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

nltk.download('punkt')
logging.basicConfig(format='%(message)s', level=logging.INFO)

DEFAULT_ALPHA = 1e-5
DEFAULT_MAX_FEATURES = None


class LyricsClassifier:
    DEFAULT_MODEL_PATH = os.getcwd() + '/models/model.pkl'
    DEFAULT_DATA_PATHS = os.getcwd() + '/models/data/train.tsv'

    def __init__(self):
        self.data_paths = self.DEFAULT_DATA_PATHS
        self.loss = 'hinge'
        self.penalty = 'l2'
        self.best_params = None
        self.tokenizer = word_tokenize

    def create_model(self, max_iter, alpha, max_features):
        """
        Initialize svm model pipeline with CountVectorizer, TfidfTransformer and SGDClassifier
        """
        self.text_clf = Pipeline([
            ('vector', CountVectorizer(tokenizer=self.tokenizer, max_features=max_features)),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss=self.loss, penalty=self.penalty, alpha=alpha, max_iter=max_iter))
        ])

    def train_model(self, data_paths=None):
        if data_paths is not None:
            self.data_paths = data_paths

        data = pd.read_csv(self.data_paths, sep='\t')
        data.dropna(inplace=True)
        data = data.sample(frac=1).reset_index(drop=True)

        def reduce_lyrics(row):
            return row[:200]

        data['lyrics'] = data.apply(lambda row: reduce_lyrics(row['lyrics']), axis=1)

        logging.info("training model")
        self.text_clf.fit(data['lyrics'], data['genre'])

    def save_model(self, model_path: str = None):
        """ Saves model under given path. If no path is provided class specific default model path is used.

        :param model_path: Path
        :return: None
        """
        model_path = self._get_model_path(model_path)
        with open(model_path, 'wb') as out:
            cPickle.dump(self.text_clf, out)
            logging.info("Model successfully dumped in {}".format(model_path))

    def calculate_cross_val_score(self, cv=10, data=None):
        """
        :param cv: Optional number of cross validations
        :param data: Data on which cross validation will be run. If not provided default data is used.
        :return: List of scores for cross validation
        """
        data = pd.read_csv(self.data_paths, sep='\t')
        data.dropna(inplace=True)
        data = data.sample(frac=1).reset_index(drop=True)

        def reduce_lyrics(row):
            return row[:200]

        data['lyrics'] = data.apply(lambda row: reduce_lyrics(row['lyrics']), axis=1)
        logging.info("Running cross validation")
        scores = cross_val_score(self.text_clf, data['lyrics'], data['genre'], cv=cv)
        logging.info("Cross validation accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        return scores

    def predict(self, sentences):
        """
        Make prediction using pretrained model
        :param sentences: List of lyrics
        :return: List od results -> ['Hip-Hop', 'Rock']
        """
        result = self.text_clf.predict(sentences)
        return result

    def load(self, filepath=None):
        """
        Loads pretrained model to use it. If no path is provided class specific default model path is used.
        :param filepath: path to the model
        """
        filepath = self._get_model_path(filepath)
        if not os.path.isfile(filepath):
            logging.info("No file was found. Building model now...")
            model = LyricsClassifier.build_uninitialized()
            model.train_model()
            model.save_model()
        with open(filepath, 'rb') as fid:
            self.text_clf = cPickle.load(fid)
        logging.info("Loaded successfully model {}".format(filepath))

    @classmethod
    def build_uninitialized(cls, max_iter=100, alpha=DEFAULT_ALPHA, max_features=DEFAULT_MAX_FEATURES):
        """ Builds the empty version of the basic model which later needs to be trained.

        :return:
        """
        model = cls()
        model.create_model(max_iter, alpha, max_features)
        return model

    @classmethod
    def build(cls, path=None):
        """ Builds the empty model and initializes it from the given or default path.

        :param path:
        :return:
        """
        path = cls._get_model_path(path)
        model = cls()
        model.load(path)
        return model

    @classmethod
    def _get_model_path(cls, model_path):
        return cls.DEFAULT_MODEL_PATH if model_path is None else model_path


def main():
    model = LyricsClassifier.build_uninitialized()
    model.train_model()
    # model.calculate_cross_val_score()
    model.save_model()

    model = LyricsClassifier.build()

    prediction = model.predict(["Hey, gimme my money"])
    logging.info(prediction)
    prediction2 = model.predict(['Love me harder. I want to be yours\nWe born to be together.'])
    logging.info(prediction2)


if __name__ == '__main__':
    main()
