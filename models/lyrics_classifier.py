import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix


class LyricsClassifier():
    def __init__(self, file='lyrics_filtered.tsv'):
        self.data = self._get_data(file)

    def _get_data(self, file):
        df = pd.read_csv(file, sep='\t')
        df.dropna(inplace=True)

        def reduce_lyrics(row):
            return row[:200]
        df['lyrics'] = df.apply(lambda row: reduce_lyrics(row['lyrics']), axis=1)
        print('Corpus: ', df.shape)
        return df

    def vectorize_lyrics(self, sentence=None):
        vectorizer = TfidfVectorizer(min_df=1)
        corpus = sentence if sentence else self.data['lyrics']
        if sentence:
            x2 = vectorizer.transform(sentence)
            return pd.SparseDataFrame(X2, columns=vectorizer.get_feature_names(), default_fill_value=0)
        return vectorizer.fit_transform(corpus)

    def train(self, x, y):
        svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3 ,random_state=42)
        svm.fit(x, y)
        return svm

    def evaluate(self):
        pass

    def run(self, sentence=None):
        print('Vectorization...')
        x = self.vectorize_lyrics()
        y = self.data['genre']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        print('Training...')
        svm = self.train(x_train, y_train)
        if sentence:
            x_new = self.vectorize_lyrics(sentence)
            import pdb
            pdb.set_trace()
            predict = svm.predict(x_new)
            print(predict.data)
        else:
            predict = svm.predict(x_test)
            print(svm.score(x_test, y_test))
            print(confusion_matrix(predict, y_test))


if __name__ == '__main__':
    l_classifier = LyricsClassifier()
    l_classifier.run(['Bitch gimme my money'])

