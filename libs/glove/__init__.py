import multiprocessing
import os.path

from ._glove import _GloVe


class GloVe(object):
    def __init__(self, window_size=10, embedding_size=32, min_word_frequency=5, threads=-1):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.min_word_frequency = min_word_frequency
        self.threads = threads
        if self.threads == -1:
            self.threads = multiprocessing.cpu_count()

        self.cpp = _GloVe(self.embedding_size, self.threads)

        self.text_files = []
        self.setted_win_weight = False

    def set_window_weight_function(self, f):
        weights = [1] + [f(float(x)) for x in range(1, self.window_size+1)]
        self.cpp.set_window_weights(weights)
        self.setted_win_weight = True

    def add_file(self, path):
        if os.path.isfile(path):
            self.text_files.append(path)
        else:
            print("Warn file '%s' doesn't exist" % path)

    def compute_dictionary(self):
        for text_file in self.text_files:
            self.cpp.add_file(text_file)

        return self.cpp.filter_dictionary(self.min_word_frequency)

    def save_dictionary(self, path):
        self.cpp.save_dictionary(path)

    def load_dictionary(self, path):
        self.cpp.load_dictionary(path)

    def compute_coocurrences(self):
        if not self.setted_win_weight:
            self.set_window_weight_function(lambda x: 1.0/x)

        for text_file in self.text_files:
            self.cpp.compute_document_vector(text_file, self.window_size)

        return self.cpp.compute_coocurrence(self.window_size)

    def save_cooccurrences(self, path):
        self.cpp.save_cooccurrences(path)

    def load_coocurrences(self, path):
        self.cpp.load_coocurrences(path)

    def shuffle_coocurrences(self):
        self.cpp.shuffle_coocurrences()

    def train_iteration(self, eta=0.05, x_max=10, learn_rate_decay=1e-3, mu=0.9, nu=0.999):
        return self.cpp.train_iteration(eta, x_max, learn_rate_decay, mu, nu)

    def save_embeddings(self, path):
        self.cpp.save_embeddings(path)
