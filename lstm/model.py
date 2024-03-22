import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Check for GPU availability
print("GPU available for TensorFlow:", tf.config.list_physical_devices('GPU'))
print("GPU available for PyTorch:", torch.cuda.is_available())

# Check the current device for TensorFlow
print("Current device for TensorFlow:", tf.test.gpu_device_name())

# Set TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Import data
df1 = pd.read_json('../Dataset/Sarcasm_Headlines_Dataset.json', lines=True)
df2 = pd.read_json('../Dataset/Sarcasm_Headlines_Dataset_v2.json', lines=True)
frames = [df1, df2]
df = pd.concat(frames)


# Tokenize to vectorize and convert texts into features
for idx, row in df.iterrows():
    row.iloc[0] = row.iloc[0].replace('rt', ' ')

max_features = 2500
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['headline'].values)
X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X)
Y = pd.get_dummies(df['is_sarcastic']).values

# Splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Building the model
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32  # You may change batch size

# Fitting the model on the training data with GPU
with tf.device('/GPU:0'):
    model.fit(X_train, Y_train, epochs=20, batch_size=batch_size)

keras_model_path = "./keras_model.h5"
# Saving the Keras model
model.save(keras_model_path)
# Load the saved Keras model
loaded_model = tf.keras.models.load_model(keras_model_path)


