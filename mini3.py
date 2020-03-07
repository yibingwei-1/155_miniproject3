import os
import numpy as np

from preprocess import (
    parse_data

)

from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission,
)

# pre-porcessing
quatrain_lists, volta_lists, couplet_lists, word_to_int, int_to_word = parse_data('data/constitution.txt')

# train
hmm = unsupervised_HMM(quatrain_lists, n_states=10, N_iters=100)

# sample naive sentence
print('Sample Naive Sentence:\n====================')
print(sample_sentence(hmm, word_to_int, n_words=10))

def get_naive_sonnet():
    sonnet = ''
    for i in range(14):
        if i % 4 ==0:
            sonnet += '\n'
        sonnet += sample_sentence(hmm, obs_map, 10) + ',\n'
    return sonnet

print('Naive Sonet:\n====================')
print (get_naive_sonnet() + '\n\n\n\n')

# Poetry Generation

