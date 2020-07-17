'''latest problem: broadcast error (unknown)'''

import tensorflow as tf
import tensorflow_addons as tfa
import json
import h5py
import os
import numpy as np
import pandas as pd
import pickle

num_words = 20000
maxlen=30
batch_size=100
unit=256
beam_width=3

def load_embeddings(path):
    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index
print("started: loading embeddings")
embeddings_index = load_embeddings('glove.6B.100d.txt')
print("finished: loading embeddings")

def extract_embeddings(vocab_size, embeddings_index):    
    embedding_dim=100
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in human_vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


print('Found %s word vectors.' % len(embeddings_index))
    

class dataset():
    #if txt bool is True, the dataset is a txt file, else, it may be csv file.
    def __init__(self, path, txt=True, csv_cols_name=None):
        self.path = path
        self.txt = txt
        self.col = csv_cols_name

    def clean_dataset(self):
        if self.txt==True:
            machine_vocab = {}
            with open(self.path,'r') as file:
                s = file.read()
            sen = s.split(".")
            sen = [x.strip() for x in sen]
            s=None
            for i in range(len(sen))[::-1]:
                words = sen[i].split()
                sen[i] = " ".join(words)

            return sen

        else:
            df = pd.read_csv(self.path, usecols=[self.col])
            sen = df[self.col]
            sen = [x.strip() for x in sen]
            for i in range(len(sen))[::-1]:
                words = sen[i].split()
                if len(words)>=3:
                    temp = " ".join(words)
                    sen[i] = '<start> ' + temp + ' <end>'
                else:
                    del sen[i]

            return sen

    def split_datset(self, clean_list):
        human_sentences=[]
        machine_sentences=[]
        for i in range(0, len(clean_list)):
            if i%2==0:
                human_sentences.append(clean_list[i])
            else:
                machine_sentences.append(clean_list[i])

        if len(human_sentences)>len(machine_sentences):
            del human_sentences[len(human_sentences)-1]
        elif len(machine_sentences)>len(human_sentences):
            del machine_sentences[len(machine_sentences)-1]
        
        return human_sentences, machine_sentences
    
    def return_vacab_and_padded(self, List, num_words, maxlen):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<OOV>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(List)
        vocab = tokenizer.word_index
        reversed_vocab = {x:y for (y,x) in vocab.items()}

        sequences = tokenizer.texts_to_sequences(List)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
        
        return padded, vocab, reversed_vocab

    def create_dataset(self, X, Y):
        BATCH_SIZE=100
        dataset = tf.data.Dataset.from_tensor_slices((X,Y))
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        return dataset

class generator(tf.keras.Model):
    def __init__(self, human_vocab, human_vocab_size, machine_vocab_size, embedding_dim):
        super().__init__()
        self.human_vocab = human_vocab
        self.hvs = human_vocab_size
        self.mvs = machine_vocab_size
        self.embedding_dim = embedding_dim
        self.matrix_human = extract_embeddings(self.hvs, embeddings_index)
        self.matrix_machine = extract_embeddings(self.mvs, embeddings_index)
        self.Eembedding = tf.keras.layers.Embedding(self.hvs, self.embedding_dim, weights=[self.matrix_human], trainable=False)
        self.ELSTM = tf.keras.layers.LSTM(unit, return_sequences=True, return_state=True)
        self.cell = tf.keras.layers.LSTMCell(unit)
        self.attention_mechanism = self.create_attention(unit, memory=None)
        self.DLSTMCell = self.wrap_attention(self.cell)
        self.Dense = tf.keras.layers.Dense(self.mvs, activation='softmax')
        self.sampler = tfa.seq2seq.ScheduledOutputTrainingSampler(
            sampling_probability=0.0000000000001,
            time_major=False,
            seed=None
        )
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.DLSTMCell,
            self.sampler,
            output_layer=self.Dense
        )
        self.Oembedding = tf.keras.layers.Embedding(self.mvs, self.embedding_dim, weights=[self.matrix_machine], trainable=False)
        self.ODense = tf.keras.layers.Dense(self.embedding_dim)

    def create_attention(self, units, memory):
        attention_mech = tfa.seq2seq.LuongMonotonicAttention(units=units, memory=memory)
        return attention_mech

    def wrap_attention(self, cell):
        Cell = tfa.seq2seq.AttentionWrapper(cell=cell, attention_mechanism=self.attention_mechanism)
        return Cell

    def call(self, Input):
        Input = tf.convert_to_tensor(Input)
        out = self.Eembedding(Input)
        out, state_a, state_c = self.ELSTM(out, initial_state=[tf.zeros((batch_size,unit)), tf.zeros((batch_size,unit))])
        self.attention_mechanism.setup_memory(out)

        initial = self.DLSTMCell.get_initial_state(batch_size=batch_size ,dtype = tf.float32)
        encoder_state = tfa.seq2seq.tile_batch([state_a, state_c], multiplier=1)
        state = initial.clone(cell_state=encoder_state)

        out = self.Oembedding(out)
        finished, next_inputs, next_state = self.decoder.initialize(out, initial_state=state)

        output=[]
        response = []
        for i in range(maxlen):
            o, next_state, next_inputs, finished = self.decoder.step(i, next_inputs, next_state)
            o=o[0]
            response.append(o)
            embeddings = self.matrix_machine
            embeddings = tf.cast(embeddings, tf.float32)
            result = tf.linalg.matmul(o, embeddings)
            concat = tf.keras.layers.Concatenate(axis=1)([result,next_inputs])
            concat = tf.keras.layers.Dense(unit)(concat)
            next_inputs = concat
            apendee = tf.reduce_mean(result, axis=0)
            output.append(apendee)

        response = tf.transpose(response, perm=[1,0,2])
        r = tf.convert_to_tensor(response)
        r = tf.argmax(r, axis=2)
        output = tf.transpose(output)
        passable = tf.convert_to_tensor(output)
        passable = tf.reshape(passable, (100,30,1))

        return passable, r

class discriminator(tf.keras.Model):
    def __init__(self, machine_vocab_size, embedding_dim):
        super().__init__()
        self.machine_vocab_size = machine_vocab_size
        self.cnn_q = tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu')
        self.cnn_r = tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu')
        self.embedding_layer = tf.keras.layers.Embedding(machine_vocab_size+1, embedding_dim)
        self.Dense_1 = tf.keras.layers.Dense(50, activation='relu')
        self.Dense_2 = tf.keras.layers.Dense(10, activation='relu')
        self.Dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def prepare_Y(self, Y):
        S = tf.split(Y, Y.shape[1], axis=1)
        prepared_Y = []
        embeddings = extract_embeddings(self.machine_vocab_size, embeddings_index)
        for i in S:
            p = tf.one_hot(i, depth=self.machine_vocab_size)
            p = tf.keras.layers.Flatten()(p)
            embeddings = tf.cast(embeddings, tf.float32)
            out = tf.linalg.matmul(p,embeddings)
            out = tf.reduce_mean(out, axis=0)
            prepared_Y.append(out)
        prepared_Y = tf.transpose(prepared_Y)
        prepared_Y = tf.convert_to_tensor(prepared_Y)
        prepared_Y = tf.reshape(prepared_Y, (100,30,1))
        return prepared_Y

    
    def call(self, Input, switch=False):
        if switch==False:
            out = self.cnn_r(Input)
            out = tf.keras.layers.GlobalAveragePooling1D()(out)
        else:
            out = self.prepare_Y(Input)
            out = self.cnn_q(out)
            out = tf.keras.layers.GlobalAveragePooling1D()(out)

        denseout = self.Dense_1(out)
        denseout = self.Dense_2(denseout)
        output = self.Dense(denseout)

        return output


class GAN():
    def __init__(self, human_vocab, human_vocab_size, machine_vocab_size):
        self.human_vocab = human_vocab
        self.human_vocab_size = human_vocab_size
        self.machine_vocab_size = machine_vocab_size
        self.embedding_dim = 100
        self.gmodel = self._create_generator_()
        self.dmodel = self._create_discriminator_()

    def _create_generator_(self):
        model = generator(self.human_vocab, self.human_vocab_size, self.machine_vocab_size, self.embedding_dim)
        return model

    def _create_discriminator_(self):
        model = discriminator(self.machine_vocab_size, self.embedding_dim)
        return model

    def generator_loss(self, disc_fake):
        gen_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.ones((batch_size,1)), y_pred=disc_fake))

        return gen_loss

    def discriminator_loss(self, disc_real, disc_fake):
        disc_loss_real = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.ones((batch_size,1)), y_pred=disc_real))
        disc_loss_fake = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.zeros((batch_size,1)), y_pred=disc_fake))

        return disc_loss_real + disc_loss_fake

    def _define_optimizers_(self, lr_gen, lr_disc):
        return tf.keras.optimizers.Adam(learning_rate=lr_gen), tf.keras.optimizers.Adam(learning_rate=lr_disc)

    def run_optimization_one(self, X, Y, lr_gen=0.001, lr_disc=0.001):

        generator = self.gmodel
        discriminator = self.dmodel
        gen_opt, disc_opt = self._define_optimizers_(lr_gen, lr_disc)
        
        with tf.GradientTape() as g:
                
            fake_response,_ = generator(X, training=True)
            disc_fake = discriminator(fake_response, training=True)
            disc_real = discriminator(Y, training=True, switch=True)
            disc_loss = self.discriminator_loss(disc_fake, disc_real)
                
        # Training Variables for each optimizer
        gradients_disc = g.gradient(disc_loss,  discriminator.trainable_variables)
        disc_opt.apply_gradients(zip(gradients_disc,  discriminator.trainable_variables))
        
        return disc_loss

    def run_optimization_two(self, X, lr_gen=0.001, lr_disc=0.001):

        generator = self.gmodel
        discriminator = self.dmodel
        gen_opt, disc_opt = self._define_optimizers_(lr_gen, lr_disc)
        
        with tf.GradientTape() as g:
                
            fake_response,_ = generator(X, training=True)
            disc_fake = discriminator(fake_response, training=True)

            gen_loss = self.generator_loss(disc_fake)
                
        gradients_gen = g.gradient(gen_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
        return gen_loss

  
'''================================================================================================================================================'''


dialogues = dataset(path="All-seasons.csv", txt=False, csv_cols_name="Line")
clean = dialogues.clean_dataset()
human_sentences, machine_sentences = dialogues.split_datset(clean)
X, human_vocab, r_human_vocab = dialogues.return_vacab_and_padded(human_sentences, num_words, maxlen)
Y, machine_vocab, r_machine_vocab = dialogues.return_vacab_and_padded(machine_sentences, num_words, maxlen)

dataset = dialogues.create_dataset(X,Y)

GAN_AEL = GAN(human_vocab, len(human_vocab), len(machine_vocab))

EPOCHS=1

print("Started: training")
for epoch in range(0, EPOCHS):
    os.system("clear")
    c=0
    i = 0
    which = True  # True -> disc, False -> gen
    disc_loss = gen_loss = -1
    for X, Y in iter(dataset):
        i += 1
        c+=1
        if c==5:
            which = not which
            c = 0
        if which:
            disc_loss = GAN_AEL.run_optimization_one(X,Y)
        else:
            gen_loss = GAN_AEL.run_optimization_two(X)
        if gen_loss<0.06 and gen_loss>0:
            break
        print('\r', "batch:", i, "gen_loss:", float(gen_loss), "disc_loss:", float(disc_loss), end='', flush=True)
    print()
    '''for debugging perposes'''
    # X,Y = next(iter(dataset))
    # gen_loss, disc_loss = GAN_AEL.run_optimization(X,Y)
    print("epoch:",epoch+1,"gen_loss:",float(gen_loss),"disc_loss:",float(disc_loss))

print("finished: training")
print("started: saving model to disk")    

model = GAN_AEL.gmodel
model.save_weights("model.h5")
print("finished: saving model to disk")

def save_dict(filename, a):
    filename+=".pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

save_dict("human_vocab", human_vocab)
save_dict("r_machine_vocab", r_machine_vocab)