from libs.glove import GloVe
import time
import math
import sys

document_path = 'data/text8'

if len(sys.argv) >= 2:
    document_path = sys.argv[1]

glove = GloVe(window_size=12, embedding_size=64)

glove.set_window_weight_function(lambda x: 1.0/math.sqrt(x))

glove.add_file(document_path)

dict_time = time.time()
print("Computing dictionary...")
word_count = glove.compute_dictionary()
print("Got %d words" % word_count)
print("Computed dictionary in %.2f seconds"%(time.time() - dict_time))

glove.save_dictionary("dictionary.txt")

cooccurrences_time = time.time()
print("Computing cooccurrences...")
cooccurences_count = glove.compute_coocurrences()
print("Got %d cooccurrences" % cooccurences_count)
print("Computed cooccurrences in %.2f seconds"%(time.time() - cooccurrences_time))

num_iterations = 15
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

glove.save_embeddings("vectors.txt")
