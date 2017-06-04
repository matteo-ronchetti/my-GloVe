import os
from libs.glove import GloVe

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def merge_obj(a, b):
    res = dict()
    for key in a:
        res[key] = b[key] if key in b else a[key]
    return res

class GloVeMultitrainer:
    default = {
        'window_size': 15,
        'eta': 0.05,
        'x_max': 10.0,
        'learn_rate_decay': 1e-3,
        'num_iterations': 15
    }

    def __init__(self, folder, embedding_size=64, min_word_frequency=10, threads=-1):
        mkdir(folder)

        self.folder = folder
        self.glove = GloVe(embedding_size=embedding_size, min_word_frequency=min_word_frequency, threads=threads)
        self.options = []

    def add_options(self, options):
        for opt in options:
            self.options.append(merge_obj(self.default, opt))

    def add_file(self, path):
        self.glove.add_file(path)

    def train(self):
        print("Computing dictionary...")
        word_count = self.glove.compute_dictionary()
        print("Got %d words" % word_count)

        dict_path = os.path.join(self.folder, "dictionary.txt")
        self.glove.save_dictionary(dict_path)

        self.options = sorted(self.options, key=lambda x: x['window_size'])
        prev_win_size = 0
        for i, opt in enumerate(self.options):
            print("Training with options:")
            print(opt)
            if opt['window_size'] != prev_win_size:
                print("Computing cooccurrences...")
                self.glove.window_size = opt['window_size']
                cooccurences_count = self.glove.compute_coocurrences()
                print("Got %d cooccurrences" % cooccurences_count)
                # self.glove.save_cooccurrences(os.path.join(my_path, "cooccurrences"))
                prev_win_size = opt['window_size']
            else:
                print("Reusing precomputed cooccurrences")

            for j in range(opt['num_iterations']):
                if j % 5 == 0:
                    self.glove.shuffle_coocurrences()
                score = self.glove.train_iteration(eta=opt['eta'], learn_rate_decay=opt['learn_rate_decay'], x_max=opt['x_max'])
                print("Iteration %d, score: %f"%(j+1, score))

            my_path = os.path.join(self.folder, str(i))
            mkdir(my_path)
            self.glove.save_embeddings(os.path.join(my_path, "embeddings.txt"))
