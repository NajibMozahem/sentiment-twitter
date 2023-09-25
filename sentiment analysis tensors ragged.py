#------------------------------------------------
# Sentiment analysis using ragged tensors to allow for unequal length sequences
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import re
import spacy

# the data used in this code can be downloaded from here:
# https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview
# make sure the proper working director is set
os.chdir("twitter sentiment")
# read both the train and test data sets
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
#we join the two data sets to fit the tokenizer on the entire data set
df = pd.concat([df_train.assign(ind='train'), df_test.assign(ind='test')], axis=0)
#keep only positive and negative sentiments
df = df[df['sentiment'] != 'neutral']
#convert sentiment to 0 and 1
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
#convert the text column to a string
df['text'] = df['text'].astype(str)
# use spacy to lemmatize the text column
nlp = spacy.load('en_core_web_sm')
# create a function to pre_process the data
def pre_process_text(text):
    # include the lemmatization
    # some people also remove the stop words, however this would result in words such as 'not' being removed, so I will just keep them
    text = " ".join([y.lemma_ for y in nlp(text) if y.is_alpha])
    text = text.lower()
    # remove spaces and punctuation
    text = re.sub('[^\w\s]', '', text)
    return text

# preprocess the 'text' column
df['text'] = df['text'].apply(pre_process_text)

# set the maximum size of the vocabulary
max_features = 3000
#create the tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features, split=' ')
# fir the tokenizer on the entire data set
tokenizer.fit_on_texts(df['text'].values)
# we now split the data set again into train and vali
df_train, df_val = df[df['ind']=='train'], df[df['ind']=='test']
# use the tokenizer to convert the training text into sequences
X_train = tokenizer.texts_to_sequences(df_train['text'].values)
#create a ragged tensor data set which will allow us to have sequences of different lengths
X_train_ragged = tf.ragged.constant(X_train)
#create a ragged tensor for the outcome
Y_train = df_train['sentiment'].values
Y_train_ragged = tf.ragged.constant(Y_train)
#Now do the same for the validation data set
X_val = tokenizer.texts_to_sequences(df_val['text'].values)
X_val_ragged = tf.ragged.constant(X_val)
Y_val = df_val['sentiment'].values
Y_val_ragged = tf.ragged.constant(Y_val)
# You can see that each tensor in the X_train_ragged data set has a different length
print(X_train_ragged)

#We now create the model. RNN allow for the input not be be of uniform size.
#The first layer is an embedding layer. This layer will create a vector of size 128 for each word in the vocabulary
#The second and third layers are LSTM layers. These layers will create a vector of size 128 for each word in the vocabulary
#The fourth layer is a dense layer with one neuron and a sigmoid activation function. This layer will output a number between 0 and 1

model = keras.models.Sequential([
    keras.layers.Input(shape=(None,), dtype=tf.int64, ragged=True),
    keras.layers.Embedding(max_features, 128),
    keras.layers.LSTM(128, dropout=0.2, return_sequences=True, recurrent_dropout=0.2),
    keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_ragged, Y_train_ragged, epochs=10, validation_data=(X_val_ragged, Y_val_ragged))

# We create a function to make predictions
def make_prediction(tweet, tokenizer, model):
    tweet = pre_process_text(tweet)
    tweet = tokenizer.texts_to_sequences([tweet])
    prediction = model.predict(tweet)
    if prediction >= 0.5:
        print("The sentiment is positive")
    else:
        print('The sentiment is negative')

#We can use this function to predict the snetiment of a tweet
tweet = 'It would have been better if you just stayed silent'
make_prediction(tweet, tokenizer, model)


