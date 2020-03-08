from keras.layers import LSTM, Input, Dense
from keras.optimizers import RMSprop
from keras import Model
import numpy as np


def train(x, y):

    input_shape = 40
    input_data = Input(shape=(input_shape,))

    x = LSTM(
        units=200,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True
    )(input_data)

    output = Dense(
        units=10,
        activation='softmax'
    )(x)

    model = Model(input_data, output)

    optimizer = RMSprop(learning_rate=0.01)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer
    )

    model.fit(
        x=x,
        y=y,
        batch_size=16,
        epochs=100
    )


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def test(x, model):
    preds = model.predict(x)



