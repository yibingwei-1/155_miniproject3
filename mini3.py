import os
import numpy as np
import random

from preprocess import (
    parse_data
)

from write_poems import (
    count_sentence_syllables,
    get_last_syllable
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
poem_lists, uatrain_lists, volta_lists, couplet_lists, word_to_int, int_to_word = parse_data('data/shakespeare.txt')

# train HMM
hmm = unsupervised_HMM(poem_lists, n_states=20, N_iters=10)

# sample naive sentence
print('Sample Naive Sentence:\n====================')
print(sample_sentence(hmm, word_to_int, n_words=10))

def write_naive_sonnet():
    sonnet = ''
    for i in range(14):
        if i % 4 ==0:
            sonnet += '\n'
        sonnet += sample_sentence(hmm, word_to_int, 10) + ',\n'
    return sonnet

print('Naive Sonet:\n====================')
print (write_naive_sonnet() + '\n\n\n\n')

# Poetry Generation

def write_rhyming_sonnet():
    sonnet = ''

    phoneme_sentences = {}
    # generate 360 sentences of length 10
    count = 0
    while count < 360:
        sentence = sample_sentence(hmm, word_to_int, n_words=10)
        num_syllables = count_sentence_syllables(sentence)
        if num_syllables == 10:
            last_syllable = get_last_syllable(sentence)
            phoneme_sentences[last_syllable] = sentence
            count += 1

    # get the structure
    structure = get_structure()
    for i,syllable in enumerate(structure):
        if i % 4 == 0:
            sonnet += '\n'
        sonnet += random.choice(phoneme_sentences[syllable]) + ',\n'
     return sonnet

print('Rhyming Sonet:\n====================')
print (write_rhyming_sonnet() + '\n\n\n\n')


