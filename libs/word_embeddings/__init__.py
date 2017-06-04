import numpy as np
import itertools
from scipy.spatial import cKDTree


def get_word_vector_len(path):
    with open(path) as f:
        line = f.readline()
        return len(line.split(" ")) - 1


class WordEmbeddings(object):
    dictionary = {}
    inverse_dictionary = {}
    frequences = {}

    def __init__(self, path, dict_path):
        print("Reading dictionary...")
        self._read_dictionary(dict_path)
        print("Reading vector embeddings...")
        self._read_vectors(path)
        print("Building kd-tree...")
        self._make_tree()

    def _read_vectors(self, path):
        self.word_vector_len = get_word_vector_len(path)
        self.vectors = np.zeros((len(self.dictionary), self.word_vector_len))
        self.normalized = np.zeros((len(self.dictionary), self.word_vector_len))

        with open(path) as f:
            for line in f:
                vals = line.rstrip().split(' ')
                word = vals[0]
                # if word in self.dictionary:
                i = self.dictionary[word]
                self.vectors[i, :] = np.array([float(x) for x in vals[1:]])
                self.normalized[i, :] = self.vectors[i, :] / np.linalg.norm(self.vectors[i, :])

    def _read_dictionary(self, dict_path):
        total_count = 0
        with open(dict_path) as f:
            for line in f:
                w, i, count = line.strip().split(" ")
                i = int(i) - 1
                self.dictionary[w] = i
                self.inverse_dictionary[i] = w
                self.frequences[w] = np.log(float(count))
                total_count += np.log(float(count))

        for w in self.frequences:
            self.frequences[w] /= total_count

    def _make_tree(self):
        self.tree = cKDTree(self.normalized)

    def frequency_weight(self, w):
        return 1.0/self.frequences[w]

    def match(self, x, num=10):
        xn = x / np.linalg.norm(x)
        _, winners = self.tree.query(xn, k=num)
        return [self.inverse_dictionary[i] for i in winners]

    def represent_string(self, txt):
        words = txt.lower().split(" ")
        x = np.zeros(self.word_vector_len)
        count = 0
        for w in words:
            if w in self.dictionary:
                idf = self.frequency_weight(w)
                x += self[w, :] * idf
                count += idf

        return x / count

    def __getitem__(self, key):
        return self.vectors[self.dictionary[key], :]

    def __contains__(self, key):
        return key in self.dictionary

    def similarity(self, a, b):
        va = self.normalized[self.dictionary[a]]
        vb = self.normalized[self.dictionary[b]]

        return 1 - np.dot(va, vb)

    # def normalize(self):
    #     norm = np.zeros(self.word_vector_len)
    #     for w in self.vectors:
    #         norm = np.maximum(norm, np.absolute(self.vectors[w]))
    #
    #     for w in self.vectors:
    #         np.divide(self.vectors[w], norm, self.vectors[w])
    #         self.normalized[w] = self.vectors[w] / np.linalg.norm(self.vectors[w])

    def analogy(self, a1, a2, b1):
        v = self[a2] - self[a1] + self[b1]
        v /= np.linalg.norm(v)
        _, winners = self.tree.query(v, 20)
        words = [self.inverse_dictionary[i] for i in winners]
        return [w for w in words if w not in [a1,a2,b1]]


def read_structured_file(path, cb=None):
    res = dict()
    key = None

    with open(path) as f:
        for line in f:
            if len(line) > 1:
                if line[0] in [' ', '\t']:
                    data = line.strip()
                    if len(data) > 0:
                        if cb:
                            res[key].append(cb(data))
                        else:
                            res[key].append(data)
                else:
                    line = line.strip()
                    if line[-1] == ':':
                        line = line[:-1]
                    key = line.strip()
                    res[key] = list()

    return res


class WordRelations:
    def __init__(self, path):
        self.relations = read_structured_file(path, lambda x: x.split(" "))

    def relation_vectors(self, embeddings):
        arrows = dict()
        for relation in self.relations:
            arrows[relation] = np.zeros(embeddings.word_vector_len)
            for pair in self.relations[relation]:
                arrows[relation] += embeddings[pair[1]] - embeddings[pair[0]]
            arrows[relation] /= len(self.relations[relation])
        return arrows

    def combinations(self):
        for relation in self.relations:
            pairs = self.relations[relation]
            for analogy in itertools.product(pairs, pairs):
                if analogy[0][0] != analogy[1][0]:
                    yield analogy[0] + analogy[1]
