# %%
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
# Read the data
df = pd.read_csv('balenced_data.csv')

# %%
# Parameters
max_features = 2000
embed_dim = 128
lstm_out = 196
batch_size = 32

# %%
# Text Preprocessing
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X)

# %%
# Convert categorical labels to numbers
le = LabelEncoder()
Y = le.fit_transform(df['Source'])

# %%
# Split into train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

# %%
# Define the LSTM model
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(df['Source'].unique()),activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# %%
# Train the model
model.fit(X_train, Y_train, epochs = 7) #, batch_size=batch_size, verbose = 2

# %%
# Evaluate the model
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("Score: %.2f" % (score))
print("Validation Accuracy: %.2f" % (acc))


