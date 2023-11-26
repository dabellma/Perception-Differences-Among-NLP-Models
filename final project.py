
from collections import defaultdict
import time
import random
import dynet as dy
import numpy as np

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"] # create an index for the unknown token
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("dist_semantics_lecture_movie_review_data/train.txt"))
w2i = defaultdict(lambda: UNK, w2i) # put unknown index on the front
dev = list(read_dataset("dist_semantics_lecture_movie_review_data/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

print(nwords)
print(len(w2i))
# print(w2i)
print(t2i)