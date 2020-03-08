from keras.layers import LSTM, Input, Dense
from keras.optimizers import RMSprop
from keras import Model
from keras.models import load_model
import numpy as np
import preprocess


def preprocess_file(file_name):
    poem_lists, quatrain_lists, volta_lists, couplet_lists, syllable_lists, word_to_int, int_to_word = preprocess.parse_data(file_name)
    window_size = 10
    vector_size = len(word_to_int)

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


def train(file_name):

    X, y, window_size, vector_size = preprocess_file(file_name)

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

    model.fit(
        x=X,
        y=y,
        batch_size=16,
        epochs=1
    )

    model.save('lstm.hdf5')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def test(file_name):

    X, y, window_size, vector_size = preprocess_file(file_name)

    model = load_model('lstm.hdf5')

    preds = model.predict(X)
    probs = sample(preds[0])

    print(probs)

    return probs


if __name__ == '__main__':
    train('data/shakespeare.txt')
    test('data/shakespeare.txt')



