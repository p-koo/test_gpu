import os, h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tfomics
from tensorflow.keras import layers, Model, Input


def cnn_dist(input_shape, num_labels, activation='relu',
          units=[24, 32, 48, 64, 96], dropout=[0.1, 0.2, 0.3, 0.4, 0.5],
          bn=[True, True, True, True, True], l2=None):

  # l2 regularization
  if l2 is not None:
    l2 = keras.regularizers.l2(l2)

  use_bias = []
  for status in bn:
    if status:
      use_bias.append(True)
    else:
      use_bias.append(False)

  # input layer
  inputs = keras.layers.Input(shape=input_shape)

  # layer 1
  nn = keras.layers.Conv1D(filters=units[0],
                           kernel_size=19,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[0],
                           padding='same',
                           kernel_regularizer=l2,
                           )(inputs)
  if bn[0]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation(activation)(nn)
  nn = keras.layers.Dropout(dropout[0])(nn)

  nn = keras.layers.Conv1D(filters=units[1],
                           kernel_size=7,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[1],
                           padding='same',
                           kernel_regularizer=l2,
                           )(nn)
  if bn[1]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(dropout[1])(nn)
  nn = keras.layers.MaxPool1D(pool_size=4)(nn)

  # layer 2
  nn = keras.layers.Conv1D(filters=units[2],
                           kernel_size=5,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[2],
                           padding='same',
                           kernel_regularizer=l2,
                           )(nn)
  if bn[2]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.MaxPool1D(pool_size=4)(nn)
  nn = keras.layers.Dropout(dropout[2])(nn)

  # layer 3
  nn = keras.layers.Conv1D(filters=units[3],
                           kernel_size=5,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[3],
                           padding='same',
                           kernel_regularizer=l2,
                           )(nn)
  if bn[3]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.MaxPool1D(pool_size=4)(nn)
  nn = keras.layers.Dropout(dropout[3])(nn)

  # layer 4 - Fully-connected
  nn = keras.layers.Flatten()(nn)
  nn = keras.layers.Dense(units[4],
                          activation=None,
                          use_bias=use_bias[4],
                          kernel_regularizer=l2,
                          )(nn)
  if bn[4]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(dropout[4])(nn)

  # Output layer
  logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
  outputs = keras.layers.Activation('sigmoid')(logits)

  # compile model
  return keras.Model(inputs=inputs, outputs=outputs)



data_path = '.' 
filepath = os.path.join(data_path, 'synthetic_code_dataset.h5')
with h5py.File(filepath, 'r') as dataset:
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.int32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.int32)
    model_test = np.array(dataset['model_test']).astype(np.float32)

model_test = model_test.transpose([0,2,1])
x_train = x_train.transpose([0,2,1])
x_valid = x_valid.transpose([0,2,1])
x_test = x_test.transpose([0,2,1])

N, L, A = x_train.shape


model = cnn_dist(input_shape=(L,A), num_labels=1, activation='exponential')

# set up optimizer and metrics
auroc = keras.metrics.AUC(curve='ROC', name='auroc')
aupr = keras.metrics.AUC(curve='PR', name='aupr')
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
model.compile(optimizer=optimizer, loss=loss, metrics=[auroc, aupr])


# early stopping callback
es_callback = keras.callbacks.EarlyStopping(monitor='val_auroc',
                                            patience=10,
                                            verbose=1,
                                            mode='max',
                                            restore_best_weights=True)
# reduce learning rate callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auroc',
                                                factor=0.2,
                                                patience=4,
                                                min_lr=1e-7,
                                                mode='max',
                                                verbose=1)

# train model
history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(x_valid, y_valid),
                    callbacks=[es_callback, reduce_lr])



# test model
model.evaluate(x_test, y_test)


# number of test sequences to analyze (set this to 500 because expintgrad takes long)

# get positive label sequences and sequence model
pos_index = np.where(y_test[:,0] == 1)[0]

num_analyze = len(pos_index)
X = x_test[pos_index[:num_analyze]]
X_model = model_test[pos_index[:num_analyze]]

# instantiate explainer class
explainer = tfomics.explain.Explainer(model, class_index=0)

# calculate attribution maps
saliency_scores = explainer.saliency_maps(X)

# reduce attribution maps to 1D scores
sal_scores = tfomics.explain.grad_times_input(X, saliency_scores)

threshold = 0.1
saliency_roc, saliency_pr = tfomics.evaluate.interpretability_performance(sal_scores, X_model, threshold)


print("%s: %.3f+/-%.3f"%('saliency auroc', np.mean(saliency_roc), np.std(saliency_roc)))
print("%s: %.3f+/-%.3f"%('saliency aupr', np.mean(saliency_pr), np.std(saliency_pr)))





