import os
import math
import numpy as np
# np.random.seed(123)
import pandas as pd
import tensorflow as tf
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.constraints import Constraint
from util import load_cubic

# python wgan-v2.py --db='oqmd' --device=0 --num_epochs=64 --batch_size=64
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 100, "the number of epochs for training")
flags.DEFINE_integer('batch_size', 256, "batch size")
flags.DEFINE_integer('lat_dim', 128, "latent noise size")
flags.DEFINE_integer('device', 0, "GPU device")
flags.DEFINE_integer('d_repeat', 5, "critic times")
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.device)

class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value
 
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)
 
    def get_config(self):
        return {'clip_value': self.clip_value}

def build_discriminator():
    init = GlorotNormal()

    #mix
    coords_inputs = Input(shape=(3,28))
    x = coords_inputs

    x = Conv1D(128,1,1)(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(512,1,1)(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(1024,1,1)(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(1024,2,1,padding='same')(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(2048,2,1)(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(4096,2,1)(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)


    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(1)(x)

    #build the discriminator
    model = Model(coords_inputs, x)
    return model

def build_generator(
 n_element=63,\
 n_spacegroup=123):
    init = GlorotNormal()
    #spacegroup label branch
    sp_inputs = Input(shape=(1,))
    sp = Embedding(n_spacegroup,64,embeddings_initializer=init)(sp_inputs)
    sp = Conv1D(96,1,kernel_initializer=init,bias_initializer=init)(sp)
    sp = Activation('relu')(sp)
    sp = Flatten()(sp)

    #element label branch
    element_inputs = Input(shape=(3,))
    # elements = Embedding(n_element,64,embeddings_initializer=init)(element_inputs)
    elements = tf.cast(element_inputs, tf.int32)
    elements = tf.gather(atom_embedding, elements)
    elements = Conv1D(128,1,kernel_initializer=init,bias_initializer=init)(elements)
    elements = Activation('relu')(elements)
    elements = Flatten()(elements)

    #latent inputs
    lat_inputs = Input(shape=(FLAGS.lat_dim,))
    gen = Dense(256,kernel_initializer=init,bias_initializer=init)(lat_inputs)

    #merge
    x = Concatenate()([sp, elements, gen])
    x = Dense(3*1*128)(x)
    x = Activation('relu')(x)
    x = Reshape((3,1,128))(x)  

    #coords outputs
    coords = Conv2DTranspose(1024,(1,2),(1,1),kernel_initializer=init,bias_initializer=init,activity_regularizer=l2(0.001),\
        kernel_regularizer=l1(0.001), bias_regularizer=l1(0.001))(x)
    coords = Activation('relu')(coords)
    coords = Dropout(0.5)(coords)

    coords = Conv2DTranspose(1024,(1,2),(1,1),kernel_initializer=init,bias_initializer=init,activity_regularizer=l2(0.001),\
        kernel_regularizer=l1(0.001), bias_regularizer=l1(0.001))(coords)
    coords = Activation('relu')(coords)
    coords = Dropout(0.5)(coords)

    coords = Conv2D(512,(1,1),(1,1),kernel_initializer=init,bias_initializer=init,activity_regularizer=l2(0.001),\
        kernel_regularizer=l1(0.001), bias_regularizer=l1(0.001))(coords)
    coords = Activation('relu')(coords)
    coords = Dropout(0.5)(coords)

    coords = Conv2D(512,(1,1),(1,1),kernel_initializer=init,bias_initializer=init,activity_regularizer=l2(0.001),\
        kernel_regularizer=l1(0.001), bias_regularizer=l1(0.001))(coords)
    coords = Activation('relu')(coords)
    coords = Dropout(0.5)(coords)

    coords = Conv2D(128,(1,1),(1,1),kernel_initializer=init,bias_initializer=init,activity_regularizer=l2(0.001),\
        kernel_regularizer=l1(0.001), bias_regularizer=l1(0.001))(coords)
    coords = Activation('relu')(coords)
    coords = Dropout(0.5)(coords)

    coords = Conv2D(1,(1,1),(1,1),kernel_initializer=init,bias_initializer=init,activity_regularizer=l2(0.001),\
        kernel_regularizer=l1(0.001), bias_regularizer=l1(0.001))(coords)
    coords = Activation('tanh')(coords)

    coords = Reshape((3,3))(coords)
        
    #lattice outputs
    lengths = Flatten()(coords)
    lengths = Dense(30)(lengths)
    lengths = Activation('relu')(lengths)

    lengths = Dense(18)(lengths)
    lengths = Activation('relu')(lengths)

    lengths = Dense(6)(lengths)
    lengths = Activation('relu')(lengths)

    lengths = Dense(1)(lengths)
    lengths = Activation('tanh')(lengths)

    model = Model([sp_inputs, element_inputs, lat_inputs], [coords, lengths])
    return model



def generate_real_samples(dataset, n_samples,n_element,n_spacegroup):
    symmetry,elements,coords,lengths,angles = dataset
    ix = np.random.choice(symmetry.shape[0], n_samples).astype(int)
    X_symmetry,X_elements,X_coords,X_lengths,X_angles = symmetry[ix],elements[ix],coords[ix],lengths[ix],angles[ix]

    return [X_symmetry,X_elements,X_coords,X_lengths,X_angles]

def generate_fake_lables(n_samples,aux_data):
    label_sp = np.random.choice(aux_data[1],n_samples,p=aux_data[-1])

    label_elements = []
    for i in range(n_samples):
        fff = np.random.choice(aux_data[0],3,replace=False)
        label_elements.append(fff)
    label_elements = np.array(label_elements)

    return [label_sp,label_elements]

def plot_history(d_hist, g_hist):
    plt.plot(d_hist)
    plt.xlabel('step (s)')
    plt.savefig('logs/d_loss-%d.png'%(FLAGS.device))
    plt.close()

    plt.plot(g_hist)
    plt.xlabel('step (s)')
    plt.savefig('logs/g_loss-%d.png'%(FLAGS.device))


#load dataset and build models
AUX_DATA, DATA = load_cubic()
n_element = AUX_DATA[0]
n_spacegroup = AUX_DATA[1]
candidate_element_comb = DATA[1]
n_discriminator = FLAGS.d_repeat

atom_embedding = np.load('data/cubic-elements-features.npy')
atom_embedding = tf.convert_to_tensor(atom_embedding, dtype=tf.float32)

d_model = build_discriminator()
g_model = build_generator(n_element,n_spacegroup)

# Define the loss functions to be used for discriminator
# This should be (fake_loss - real_loss)
def discriminator_loss(real_spl, fake_spl):
    real_loss = tf.reduce_mean(real_spl)
    fake_loss = tf.reduce_mean(fake_spl)
    # tf.print(fake_loss, real_loss)
    return fake_loss - real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_spl):
    return -tf.reduce_mean(fake_spl)


bacc = tf.keras.metrics.BinaryAccuracy()
cacc = tf.keras.metrics.CategoricalAccuracy()

d_optimizer = Adam(lr=0.00001, beta_1=0.5, beta_2=0.9)
g_optimizer = Adam(lr=0.00001)

#-------------
#credits to https://keras.io/examples/generative/wgan_gp/
#-------------
def gradient_penalty(batch_size, real_samples, fake_samples):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # get the interplated image
    alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_samples - real_samples
    interpolated = real_samples + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        # c = interpolated[:,:3,:]
        # l = interpolated[:,3,:]
        # a = interpolated[:,4,:]
        pred = d_model(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calcuate the norm of the gradients
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp



@tf.function
def train_step(X_real):
    real_crystal = [tf.cast(tensor, tf.float32) for tensor in X_real]
    sp_onehot = tf.one_hot(tf.cast(real_crystal[0], tf.int32),3)
    sp_onehot = tf.reshape(sp_onehot,(FLAGS.batch_size,3,1))
    e = real_crystal[1]
    e = tf.cast(e, tf.int32)
    e = tf.gather(atom_embedding, e)
    coords = real_crystal[2]
    l = tf.reshape(real_crystal[3],(FLAGS.batch_size,1,1))
    l = tf.repeat(l,3,axis=1)
    real_crystal = tf.concat([coords,e,l,sp_onehot], axis=-1)

    for _ in range(n_discriminator):
        noise = tf.random.normal(shape=(FLAGS.batch_size, FLAGS.lat_dim))
        fake_labels = generate_fake_lables(FLAGS.batch_size, AUX_DATA)
        with tf.GradientTape() as tape:
            #generate fake lattices and coords for the crystals
            fake_crystal = g_model(fake_labels+[noise],training=True)
            #get the logits for fake cystals
            sp_onehot = tf.one_hot(tf.cast(fake_labels[0], tf.int32),3)
            sp_onehot = tf.reshape(sp_onehot,(FLAGS.batch_size,3,1))
            e = fake_labels[1]
            e = tf.cast(e, tf.int32)
            e = tf.gather(atom_embedding, e)
            coords = fake_crystal[0]
            l = tf.reshape(fake_crystal[1],(FLAGS.batch_size,1,1))
            l = tf.repeat(l,3,axis=1)
            fake_crystal = tf.concat([coords,e,l,sp_onehot], axis=-1)

            fake_logits = d_model(fake_crystal,training=True)
            #get the logits for real crystals
            real_logits = d_model(real_crystal,training=True)
            #get the discriminator loss
            d_loss = discriminator_loss(real_logits, fake_logits)

            gp = gradient_penalty(FLAGS.batch_size,real_crystal,fake_crystal)
            d_loss = d_loss + 10.0*gp
        d_gradient = tape.gradient(d_loss, d_model.trainable_variables)
        d_optimizer.apply_gradients(
                zip(d_gradient, d_model.trainable_variables)
            )


    noise = tf.random.normal(shape=(FLAGS.batch_size, FLAGS.lat_dim))
    fake_labels = generate_fake_lables(FLAGS.batch_size, AUX_DATA)
    with tf.GradientTape() as tape:
        #generate fake crystal using the generator
        fake_crystal = g_model(fake_labels+[noise],training=True)
        #get the logits for fake crystal
        sp_onehot = tf.one_hot(tf.cast(fake_labels[0], tf.int32),3)
        sp_onehot = tf.reshape(sp_onehot,(FLAGS.batch_size,3,1))
        e = fake_labels[1]
        e = tf.cast(e, tf.int32)
        e = tf.gather(atom_embedding, e)
        coords = fake_crystal[0]
        l = tf.reshape(fake_crystal[1],(FLAGS.batch_size,1,1))
        l = tf.repeat(l,3,axis=1)
        fake_crystal = tf.concat([coords,e,l,sp_onehot], axis=-1)
        gen_crystal_logits = d_model(fake_crystal,training=True)
        #get the generator loss
        g_loss = generator_loss(gen_crystal_logits)
    gen_gradient = tape.gradient(g_loss, g_model.trainable_variables)
    g_optimizer.apply_gradients(
            zip(gen_gradient, g_model.trainable_variables)
        )


    return d_loss, g_loss


d_hist, g_hist = [],[]
bat_per_epo = int(DATA[0].shape[0] / FLAGS.batch_size)
for i in range(FLAGS.num_epochs*bat_per_epo):
    X_real = generate_real_samples(DATA,FLAGS.batch_size,n_element,n_spacegroup)
    d_loss, g_loss = train_step(X_real)
    print('>%d/%d, d=%.4f g=%.4f' %
            (i+1, FLAGS.num_epochs*bat_per_epo, d_loss, g_loss))

    if i%50==0:
        d_hist.append(d_loss)
        g_hist.append(g_loss)



plot_history(d_hist, g_hist)
g_model.save('models/clean-wgan-generator-%d.h5'%(FLAGS.device))
d_model.save('models/clean-wgan-discriminator-%d.h5'%(FLAGS.device))








