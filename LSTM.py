from keras.layers import LSTM, Input, Dense
from keras.optimizers import RMSprop
from keras import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import write_poems
import numpy as np
import preprocess


class LSTMGenerator(object):

    def __init__(self):
        self.word_to_int = {}
        self.int_to_word = {}

    def get_word_to_int(self):
        return self.word_to_int

    def preprocess_file(self, file_name):
        poem_lists, quatrain_lists, volta_lists, couplet_lists, self.word_to_int, self.int_to_word = preprocess.parse_data(file_name)
        window_size = 10
        vector_size = len(self.word_to_int)

        total_sequence = []
        for poem in poem_lists:
            total_sequence.extend(poem)

        X_raw = []
        y_raw = []
        for i in range(0, len(total_sequence) - window_size):
            X_raw.append(total_sequence[i: i + window_size])
            y_raw.append(total_sequence[i + window_size])

        X = np.zeros((len(total_sequence) - window_size, window_size, vector_size))
        y = np.zeros((len(total_sequence) - window_size, vector_size))

        for i in range(0, len(total_sequence) - window_size):
            y[i, y_raw[i]] = 1
            for j in range(0, window_size):
                X[i, j, X_raw[i][j]] = 1

        return X, y, window_size, vector_size

    def train(self, file_name):

        X, y, window_size, vector_size = self.preprocess_file(file_name)

        input_data = Input(shape=(window_size, vector_size,))

        lstm = LSTM(
            units=200,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True
        )(input_data)

        output = Dense(
            units=vector_size,
            activation='softmax'
        )(lstm)

        model = Model(input_data, output)

        optimizer = RMSprop(learning_rate=0.01)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer
        )

        callbacks = [
            ModelCheckpoint(
                'lstm_best.hdf5',
                monitor='loss',
                verbose=1,
                save_best_only=True,
                mode='auto',
                period=1
            ),
            EarlyStopping(
                monitor='loss',
                patience=10,
                mode='auto',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss',
                patience=5,
                verbose=1,
                mode='auto',
                factor=0.1
            )
        ]

        model.fit(
            x=X,
            y=y,
            batch_size=16,
            epochs=100,
            callbacks=callbacks
        )

        model.save('lstm.hdf5')

    @staticmethod
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        return np.argmax(probs)

    def test(self, file_name):

        X, y, window_size, vector_size = self.preprocess_file(file_name)

        model = load_model('lstm.hdf5')

        preds = model.predict(X)
        probs = []

        for pred in preds:
            probs.append(self.sample(pred))

        return probs

    def sample_sentences(self, file_name):
        probs = self.test(file_name)
        sentences = []

        for i in range(0, int(len(probs) / 10)):
            prob = probs[i: i + 10]
            sentence = []
            for j in range(0, 10):
                sentence.append(self.int_to_word[prob[j]])
            sentences.append(' '.join(sentence))

        return sentences


if __name__ == '__main__':

    LSTM_Generator = LSTMGenerator()

    LSTM_Generator.train('data/shakespeare.txt')
    # sample_sentence = LSTMGenerator.sample_sentences(LSTM_Generator, 'data/shakespeare.txt')
    # syllable_dict = preprocess.syllables_interpreter('data/Syllable_dictionary.txt', LSTM_Generator.get_word_to_int())
    #
    # for sentence in sample_sentence:
    #     print(write_poems.truncate_sentence(sentence, LSTM_Generator.get_word_to_int(), syllable_dict))
    #
    # print(sample_sentence)


