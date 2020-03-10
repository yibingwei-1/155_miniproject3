import LSTM
import preprocess
import write_poems


LSTM_Generator = LSTM.LSTMGenerator()

# LSTM_Generator.train('data/shakespeare.txt')
sample_sentence = LSTM.LSTMGenerator.sample_sentences(LSTM_Generator, 'data/shakespeare.txt')
syllable_dict = preprocess.syllables_interpreter('data/Syllable_dictionary.txt', LSTM_Generator.get_word_to_int())

for sentence in sample_sentence:
    print(write_poems.truncate_sentence(sentence, LSTM_Generator.get_word_to_int(), syllable_dict))

# print(sample_sentence)