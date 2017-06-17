from libs.word_embeddings import WordEmbeddings
from builtins import input

import argparse


parser = argparse.ArgumentParser(description='This is an utility to play with word embeddings')
parser.add_argument('embeddings', help="The path of the embeddings")

args = parser.parse_args()

embeddings = WordEmbeddings(args.embeddings)

print("Insert a word to get its nearest neighbors")
print("Insert 3 words to get the analogy completion (for example 'man woman king' returns 'queen')")
print("Just press enter to exit")


while True:
    txt = input("Query: ")
    if txt == "":
        break
    words = txt.split(" ")
    if len(words) == 3:
        res = embeddings.analogy(words[0], words[1], words[2])
        print("Completing analogy  %s : %s = %s : %s"%(words[0], words[1], words[2], res[0]))
        print("   ".join(res))
    else:
        print("Closest words to %s"%words[0])
        print("   ".join(embeddings.match(embeddings[words[0]])))
