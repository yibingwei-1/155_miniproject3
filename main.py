import LSTM
import preprocess
import write_poems
import random


LSTM_Generator = LSTM.LSTMGenerator()

# LSTM_Generator.train('data/shakespeare.txt')
sample_sentence = LSTM.LSTMGenerator.sample_sentences(LSTM_Generator, 'data/shakespeare.txt')
sentence_dict = write_poems.classify_sentence(sample_sentence)
syllable_template = write_poems.read_syllable_template()

random_syllable_template = random.choice(syllable_template)
print(random_syllable_template)

sonnet = []
for syllable in random_syllable_template:
    sonnet.append(random.choice(sentence_dict[syllable]))

sonnet = write_poems.format_poem(sonnet)
for s in sonnet:
    print(s)

