import os
import numpy as np
import random
import extract_end_syllable
from collections import defaultdict

from preprocess import (
    parse_data,
    syllables_interpreter
)

from write_poems import (
    count_sentence_syllables,
    get_last_syllable,
    truncate_sentence,
    read_syllable_template,
    read_word_syllable
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
poem_lists, uatrain_lists, volta_lists, couplet_lists, word_to_int, int_to_word = parse_data('./data/shakespeare.txt')

# train
hmm = unsupervised_HMM(poem_lists, n_states=30, N_iters=50)

# sample naive sentence
print('Sample Naive Sentence:\n====================')
print(sample_sentence(hmm, word_to_int, n_words=10))
print('\n\n\n\n')


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

def write_rhyming_sonnet(word_to_int):
    sonnet = ''
    phoneme_sentences = {}
    syllable_lists = read_syllable_template()
    syllables_dict = syllables_interpreter('./data/Syllable_dictionary.txt', word_to_int)
    word_syllable_dict = read_word_syllable()
    # generate 360 sentences of length 10
    count = 0
    while count < 1000:
        sentence = sample_sentence(hmm, word_to_int, n_words=10)

        num_syllables = count_sentence_syllables(sentence, word_to_int, syllables_dict)
        if num_syllables == 10:
            last_syllable = get_last_syllable(sentence, word_syllable_dict)
            if last_syllable not in phoneme_sentences:
                phoneme_sentences[last_syllable] = [sentence.capitalize()]
            else:
                phoneme_sentences[last_syllable].append(sentence.capitalize())
            count += 1

    # get the structure
    while True:
        structure = random.choice(syllable_lists)
        for i,syllable in enumerate(structure):
            if i % 4 == 0:
                sonnet += '\n'
            if not phoneme_sentences[syllable]:
                continue
            sentence = random.choice(phoneme_sentences[syllable])
            phoneme_sentences[syllable].remove(sentence)
            sonnet += sentence + ',\n'
        return sonnet

print('Rhyming Sonet:\n====================')
print(write_rhyming_sonnet(word_to_int) + '\n\n\n\n')


