import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_notes, config):
    sequence_input = layers.Input(shape=input_shape, dtype='float32')
    masked_input = layers.Masking(mask_value=-1.)(sequence_input)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_input)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)

    lstm_out_1 = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, activation='tanh', kernel_initializer='normal'))(x)
    dropout_1 = layers.Dropout(config.dropout)(lstm_out_1)

    for _ in range(config.num_layers-1):
        lstm_out_1 = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, activation='tanh', kernel_initializer='normal'))(dropout_1)
        dropout_1 = layers.Dropout(config.dropout)(lstm_out_1)
    
    output = layers.TimeDistributed(layers.Dense(num_notes, activation='sigmoid', kernel_initializer='normal'))(dropout_1)

    model = keras.Model(inputs=sequence_input, outputs=output)
    model.summary()
    return model