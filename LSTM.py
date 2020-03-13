from keras.layers import LSTM, Input, Dense, Lambda
from keras.optimizers import RMSprop, Adam
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

        self.char_to_int = {}
        self.int_to_char = {}

        self.window_size = 40
        self.vector_size = 0

    def get_word_to_int(self):
        return self.word_to_int

    def preprocess_file(self, file_name):
        # poem_lists, quatrain_lists, volta_lists, couplet_lists, self.word_to_int, self.int_to_word = preprocess.parse_data(file_name)
        self.window_size = 40

        pure_sonnets_file = open(file_name)
        pure_sonnets = pure_sonnets_file.read()
        pure_sonnets_file.close()

        characters = sorted(list(set(pure_sonnets)))
        self.vector_size = len(characters)

        for i, ch in enumerate(characters):
            self.int_to_char[i] = ch
            self.char_to_int[ch] = i

        X_raw = []
        y_raw = []

        for i in range(0, len(pure_sonnets) - self.window_size):
            X_raw.append([self.char_to_int[ch] for ch in pure_sonnets[i: i + self.window_size]])
            y_raw.append(self.char_to_int[pure_sonnets[i + self.window_size]])

        X = np.zeros((len(X_raw), self.window_size, len(characters)))
        y = np.zeros((len(X_raw), len(characters)))

        for i in range(0, len(X_raw)):
            y[i, y_raw[i]] = 1
            for j in range(0, self.window_size):
                X[i, j, X_raw[i][j]] = 1

        return X, y

    def train(self, file_name):

        X, y = self.preprocess_file(file_name)

        input_data = Input(shape=(self.window_size, self.vector_size,))

        lstm = LSTM(
            units=150,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True
        )(input_data)

        output = Dense(
            units=self.vector_size,
            activation='softmax'
        )(lstm)

        model = Model(input_data, output)

        optimizer = Adam(learning_rate=0.01)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        model.summary()

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
                patience=5,
                mode='auto',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss',
                patience=3,
                verbose=1,
                mode='auto',
                factor=0.1
            )
        ]

        model.fit(
            x=X,
            y=y,
            shuffle=True,
            batch_size=32,
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

    def sample_sentences(self, file_name, seed="shall i compare thee to a summer's day?\n"):
        X, y = self.preprocess_file(file_name)

        input_data = Input(shape=(self.window_size, self.vector_size,))

        lstm = LSTM(
            units=150,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True
        )(input_data)

        lstm = Lambda(
            lambda x: x / 0.75
        )(lstm)

        output = Dense(
            units=self.vector_size,
            activation='softmax'
        )(lstm)

        model = Model(input_data, output)
        model.load_weights('lstm_best_model_weights.hdf5')

        sentences = ''
        # model = load_model('lstm_best.hdf5')

        X = np.zeros((1, self.window_size, self.vector_size))
        for i, ch in enumerate(seed):
            X[0, i, self.char_to_int[ch]] = 1

        for _ in range(1000):
            preds = model.predict(X)
            prob = np.argmax(preds)

            sentences += self.int_to_char[prob]
            new_X = np.zeros((1, 1, self.vector_size))
            new_X[0, 0, prob] = 1
            X = np.concatenate((X[:, 1:, :], new_X), axis=1)

        print(sentences)
        return sentences


if __name__ == '__main__':

    LSTM_Generator = LSTMGenerator()

    # LSTM_Generator.train('data/pure_shakespeare.txt')
    sample_sentence = LSTMGenerator.sample_sentences(LSTM_Generator, 'data/pure_shakespeare.txt')
    # syllable_dict = preprocess.syllables_interpreter('data/Syllable_dictionary.txt', LSTM_Generator.get_word_to_int())
    #
    # for sentence in sample_sentence:
    #     print(write_poems.truncate_sentence(sentence, LSTM_Generator.get_word_to_int(), syllable_dict))
    #
    # print(sample_sentence)
