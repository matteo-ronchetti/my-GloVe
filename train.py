from __future__ import print_function

from libs.glove import GloVe
import time
import math
import sys
import os
import argparse


parser = argparse.ArgumentParser(description='This is an utility to train GloVe embeddings')
parser.add_argument('document_path', help="the path of the document used to compute embeddings")

parser.add_argument("-o", "--output", help="output name", default="embeddings")
parser.add_argument("-w", "--window_size", type=int,
                    help="Size of the cooccurrence window", default=10)
parser.add_argument("-i", "--iterations", type=int,
                    help="Number of iterations", default=15)
parser.add_argument("-s", "--embedding_size", type=int,
                    help="Size of the embeddings (must be a power of two)", default=32)
parser.add_argument("-m", "--min_word_frequency", type=int,
                    help="Minimum word frequency", default=5)
parser.add_argument("-t", "--threads", type=int,
                    help="Number of threads (-1 means automatic choice)", default=-1)
parser.add_argument('--sqrt-weight',action='store_true', help="Use sqrt window weights")


args = parser.parse_args()

if not os.path.isdir(args.output):
    os.mkdir(args.output)


glove = GloVe(window_size=args.window_size, embedding_size=args.embedding_size, min_word_frequency=args.min_word_frequency, threads=args.threads)


if args.sqrt_weight:
    glove.set_window_weight_function(lambda x: 1.0/math.sqrt(x))

glove.add_file(args.document_path)

dict_time = time.time()
print("Computing dictionary...")
word_count = glove.compute_dictionary()
print("Got %d words" % word_count)
print("Computed dictionary in %.2f seconds"%(time.time() - dict_time))

glove.save_dictionary(os.path.join(args.output,"dictionary.txt"))

cooccurrences_time = time.time()
print("Computing cooccurrences...")
cooccurences_count = glove.compute_coocurrences()
print("Got %d cooccurrences" % cooccurences_count)
print("Computed cooccurrences in %.2f seconds"%(time.time() - cooccurrences_time))

num_iterations = args.iterations
iter_total_time = 0
for i in range(num_iterations):
    iter_time_s = time.time()
    if i and i % 10 == 0:
        glove.shuffle_coocurrences()
    score = glove.train_iteration()
    iter_time = time.time() - iter_time_s
    iter_total_time += iter_time
    print("Iteration %d, score: %f, time: %.2f second"%(i+1, score, iter_time))

print("Did %d iterations in %.2f seconds (%.2f seconds/iteration)"%(num_iterations, iter_total_time, iter_total_time/num_iterations))

glove.save_embeddings(os.path.join(args.output,"embeddings.txt"))
print("Saved embeddings")
