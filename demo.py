from libs.word_embeddings import WordEmbeddings

embeddings = WordEmbeddings("vectors.txt", "dictionary.txt")


def print_closest(w):
    if w in embeddings:
        print("Closest words to '%s':" % w)
        print("     ".join(embeddings.match(embeddings[w], num=30)[1:]))


def print_analogy(a1, a2, b1):
    if a1 in embeddings and a2 in embeddings and b1 in embeddings:
        print("%s : %s = %s : %s" % (a1, a2, b1, embeddings.analogy(a1, a2, b1)[0]))


print_analogy("rome", "italy", "paris")
print_analogy("rome", "italy", "berlin")
print_analogy("rome", "italy", "london")
print_analogy("rome", "italy", "stockholm")
print_analogy("rome", "italy", "algiers")
print_analogy("rome", "italy", "madrid")

print("")
print_closest("italy")
print_closest("usa")
