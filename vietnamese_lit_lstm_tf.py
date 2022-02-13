# Tensorflow citation
"""
@misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}
"""

# This script follows tutorials on 
# https://www.tensorflow.org/text/tutorials/text_generation

import numpy as np 
import os
import tensorflow as tf
import time

# ### Data Source
# ### https://www.kaggle.com/sanglequang/van-hoc-viet-nam

# list current directory os.listdir(DIR): os.getcwd()
DIR = "./con-hoang_HBC/"
valid_file = ['.txt']

count = 0
for text in os.listdir(DIR):
    ext = os.path.splitext(text)[1]
    if ext.lower() in valid_file:
        if len(doc_sum) ==0:
            doc_sum = open(os.path.join(DIR,text)).read()
        if len(doc_sum) >0:
            new = open(os.path.join(DIR,text)).read()
            doc_sum = doc_sum+ " "+new
            
        count +=2
        if count>0:
            break

# sorted unique characters            
vocab = sorted(set(doc_sum))

# generate characters
chars = tf.strings.unicode_split(doc_sum, input_encoding='UTF-8')

# create ids for each characters
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(),
    invert=True, mask_token=None)

chars = chars_from_ids(ids)

# A function to join word back basing on ids
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
all_ids = ids_from_chars(tf.strings.unicode_split(doc_sum, 'UTF-8'))

# create tensor dataset
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# to split training/test data
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
  
dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# create model
model = Sequential()
model.add(Embedding(len(ids_from_chars.get_vocabulary()), embedding_dim))# Your Embedding Layer)
model.add(Bidirectional(LSTM(150, return_sequences = True)))# An LSTM Layer)
model.add(Dropout(0.2))# A dropout layer)

model.add(Dense(len(ids_from_chars.get_vocabulary())))# A Dense Layer)
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])# Pick a loss function and an optimizer)


print(model.summary())

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss,
              optimizer='adam', metrics=['accuracy'])# Pick a loss function and an optimizer)

# create custom callback
class custom_callback(tf.keras.callbacks.Callback):
    def check_accuracy(self, epoch, logs={}):
    
    #Stop training after 90 percent accuracy

        if(logs.get('accuracy') > 0.9):

          # Stop if threshold is met
            print("\nAccuracy is higher than 90 percent so cancelling training!")
            self.model.stop_training = True

callbacks = custom_callback()


# train the model
EPOCHS=50
history = model.fit(dataset, epochs=EPOCHS, callbacks=callbacks)

# based on https://www.tensorflow.org/text/tutorials/text_generation
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits = self.model(inputs=input_ids)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars
    
    
 
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)



next_char = tf.constant(['tu sang den toi '])
result = [next_char]

for n in range(25):
    next_char = one_step_model.generate_one_step(next_char)
    result.append(next_char)

result = tf.strings.join(result)

print(result[0].numpy().decode('utf-8'), '\n' + '_'*80)

# The result is not that good
# Here is a sample of the output:
"""
tu sang den toi sẽààmợ
rrứ 
"""








