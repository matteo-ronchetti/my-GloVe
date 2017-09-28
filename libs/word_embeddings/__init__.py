import os
import math
import numpy as np
from scipy.spatial import cKDTree


def get_word_vector_len(path):
    with open(path) as f:
        line = f.readline()
        return len(line.split(" ")) - 1


class WordEmbeddings(object):
    dictionary = {}
    inverse_dictionary = {}
    frequences = {}

    def __init__(self, path):
        vector_path = os.path.join(path, "embeddings.txt")
        dict_path = os.path.join(path, "dictionary.txt")

        print("Reading dictionary...")
        self._read_dictionary(dict_path)
        print("Reading vector embeddings...")
        self._read_vectors(vector_path)
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

    def match(self, x, num=20):
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

    def analogy(self, a1, a2, b1, num=20):
        v = self[a2] - self[a1] + self[b1]
        v /= np.linalg.norm(v)
        _, winners = self.tree.query(v, num)
        words = [self.inverse_dictionary[i] for i in winners]
        return [w for w in words if w not in [a1, a2, b1]]


class SentenceEmbedder(object):
    def __init__(self, word_embeddings, L=0):
        self.word_embeddings = word_embeddings
        self.L = L

    @staticmethod
    def triangular_weights(size, pos):
        cx = pos + 0.5
        h = 2.0 / size
        ms = h / cx if cx else 0
        me = h / (cx - size) if (cx-size) else 0
        cxh = h + (0.5 * me - 0.5 * ms) / 4

        return [ms * (x + 0.5) for x in range(int(math.floor(pos)))] + [cxh] + [me * (x - size + 0.5) for x in range(int(math.floor(pos))+1, size)]

    @staticmethod
    def _represent(embeddings, weights_a, weights_b=None):
        res = np.zeros(len(embeddings[0]))
        if weights_b:
            for i in range(len(embeddings)):
                res += embeddings[i] * weights_a[i] * weights_b[i]
        else:
            for i in range(len(embeddings)):
                res += embeddings[i] * weights_a[i]

        return res

    def __call__(self, txt):
        res = []
        words = [w for w in txt.split(" ") if w in self.word_embeddings]
        embeddings = [self.word_embeddings[w] for w in words]

        idfs = [self.word_embeddings.frequency_weight(w) for w in words]
        tot_freq = sum(idfs)
        idfs = [x / tot_freq for x in idfs]

        res.append(self._represent(embeddings, idfs))
        for j in range(0, self.L):
            x = len(words) * (j + 0.5) / self.L
            res.append(self._represent(embeddings, idfs, self.triangular_weights(len(words), x)))

        embed = np.concatenate(res)
        embed /= np.linalg.norm(embed)
        return embed


class SentenceMatcher(object):
    def __init__(self, sentences, embedder):
        self.sentences = sentences
        self.embedder = embedder

        self.embedded_sentences = np.array([self.embedder(sent) for sent in sentences])
        self.tree = cKDTree(self.embedded_sentences)

    def __call__(self, txt, num=10, scores=False):
        query = self.embedder(txt)
        _, winners = self.tree.query(query, k=num)
        if scores:
            return winners, [(1+np.dot(query, self.embedded_sentences[i]))/2 for i in winners]
        return winners

