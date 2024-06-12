# %%
"""
# FCC Classifier Training
If Train_classifier parameter is 0, then this cell only shows the performance of the existing classifier on test set.
"""

# %%
import json
import time
import pandas as pd
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform, glorot_normal, he_normal
from keras.optimizers import Adamax, RMSprop, SGD, Adadelta, Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def train(treinar,train_x, train_y, validation_x, validation_y, EPOCH, class_weights, model_info, file_name, num_classes, batch_size=None,
          load_path=None):
    """
    Standard neural network training procedure.
    """
    NUM_OF_BYTE_IN_FLOW_SEQUENCE = train_x.shape[1]

    if load_path != None:
        model = load_model(load_path)  # + '_' + CLASSIFICATION_TARGET)
        print(model.summary())
        print("Model Loaded!!!")
        return model
    if model_info['Num of blocks'] != 0:
        input_shape = (NUM_OF_BYTE_IN_FLOW_SEQUENCE, 1)
        train_x = np.expand_dims(np.array(train_x), axis=2)
        validation_x = np.expand_dims(np.array(validation_x), axis=2)
    else:
        input_shape = (NUM_OF_BYTE_IN_FLOW_SEQUENCE,)
        train_x = np.array(train_x)
        validation_x = np.array(validation_x)

    train_y = to_categorical(train_y, num_classes=num_classes)
    validation_y = to_categorical(validation_y, num_classes=num_classes)

    model = Sequential()

    for i in range(model_info['Num of blocks']):
        for j in range(model_info['Num of conv layer in block']):
            conv_index = i * model_info['Num of conv layer in block'] + j
            if conv_index == 0:
                model.add(Conv1D(filters=model_info['Conv_Filter_num'][conv_index],
                                 kernel_size=model_info['Conv_Kernel_size'][conv_index], input_shape=input_shape,
                                 strides=model_info['Conv_stride_size'][conv_index], padding=model_info['Conv padding'],
                                 name='block' + str(i + 1) + '_conv' + str(j + 1)))
            else:
                model.add(Conv1D(filters=model_info['Conv_Filter_num'][conv_index],
                                 kernel_size=model_info['Conv_Kernel_size'][conv_index],
                                 strides=model_info['Conv_stride_size'][conv_index], padding=model_info['Conv padding'],
                                 name='block' + str(i + 1) + '_conv' + str(j + 1)))
            if model_info['Conv_Batch normalization'] == 'yes' and model_info[
                'Conv_Batch normalization place'] == 'before activation':
                model.add(BatchNormalization(axis=-1, name='block' + str(i + 1) + '_BN' + str(j + 1)))
            if model_info['Conv_Activations'] == 'relu':
                model.add(Activation('relu', name='block' + str(i + 1) + '_act' + str(j + 1)))
            if model_info['Conv_Activations'] == 'elu':
                model.add(ELU(alpha=1.0, name='block' + str(i + 1) + '_act' + str(j + 1)))
            if model_info['Conv_Batch normalization'] == 'yes' and model_info[
                'Conv_Batch normalization place'] == 'after activation':
                model.add(BatchNormalization(axis=-1, name='block' + str(i + 1) + '_BN' + str(j + 1)))
            if model_info['Conv_Dropout'] == 'yes':
                if model_info['Conv_Dropout rate'][conv_index] >= 1.:
                    print("Bad Dropout rate", model_info['Conv_Dropout rate'])
                    quit()
                model.add(Dropout(model_info['Conv_Dropout rate'][conv_index],
                                  name='block' + str(i + 1) + '_dropout' + str(j + 1)))
        if model_info['Conv_MaxPooling1D'] == 'yes':
            model.add(MaxPooling1D(pool_size=model_info['Pool_size'][i], strides=model_info['Pool_stride_size'][i],
                                   padding=model_info['Conv_MaxPooling1D padding'],
                                   name='block' + str(i + 1) + '_pool'))
    if model_info['Num of blocks'] != 0:
        model.add(Flatten(name='flatten'))
    for i in range(model_info['Num of fully connected layers']):
        if model_info['Num of blocks'] == 0:
            model.add(Dense(model_info['FC_Num of neurons in hidden layers'][i], kernel_initializer=he_normal(),
                            input_shape=input_shape, name='fc' + str(i + 1)))
        else:
            model.add(Dense(model_info['FC_Num of neurons in hidden layers'][i], kernel_initializer=he_normal(),
                            name='fc' + str(i + 1)))
        if model_info['FC_Batch normalization'] == 'yes' and model_info[
            'FC_Batch normalization place'] == 'before activation':
            model.add(BatchNormalization(name='fc' + str(i + 1) + '_BN'))
        if model_info['FC_Activations'] == 'relu':
            model.add(Activation('relu', name='fc' + str(i + 1) + '_act'))
        if model_info['FC_Activations'] == 'elu':
            model.add(ELU(alpha=1.0, name='fc' + str(i + 1) + '_act'))
        if model_info['FC_Batch normalization'] == 'yes' and model_info[
            'FC_Batch normalization place'] == 'after activation':
            model.add(BatchNormalization(name='fc' + str(i + 1) + '_BN'))
        if model_info['FC_Dropout'] == 'yes':
            if model_info['FC_Dropout rate'][i] >= 1.:
                print("Bad Dropout rate", model_info['FC_Dropout rate'])
                quit()
            model.add(Dropout(model_info['FC_Dropout rate'][i], name='fc' + str(i + 1) + '_dropout'))
    #model.add(Dense(num_classes[0], kernel_initializer=he_normal(), name='fc' + str(i + 2)))
    model.add(Dense(num_classes, kernel_initializer=he_normal(), name='fc' + str(i + 2)))
    model.add(Activation('softmax', name="softmax"))
    print("lr", model_info["Learning rate"])
    if model_info['Optimization'] == 'Adamax':
        OPTIMIZER = Adamax(lr=model_info["Learning rate"])
    if model_info['Optimization'] == 'SGD':
        OPTIMIZER = SGD(lr=model_info["Learning rate"])
    if model_info['Optimization'] == 'RMSprop':
        OPTIMIZER = RMSprop(lr=model_info["Learning rate"])
    if model_info['Optimization'] == 'Adadelta':
        OPTIMIZER = Adadelta(lr=model_info["Learning rate"])
    if model_info['Optimization'] == 'Adam':
        OPTIMIZER = Adam(lr=model_info["Learning rate"])

    model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER,
                  metrics=["accuracy"])
    print("Model compiled")
    print(model.summary())


    if treinar == 1: 
        best_model_save = ModelCheckpoint(file_name, monitor='val_accuracy', verbose=0, save_best_only=True,
                                          save_weights_only=True, mode='max', period=1)
        history = model.fit(train_x, train_y,
                            batch_size=batch_size, epochs=EPOCH, verbose=1, callbacks=[best_model_save], shuffle=True,
                            validation_data=(validation_x, validation_y), class_weight=class_weights)
        print("Load Best Model...")
        #model.load_weights('testeClassifiers/Trained_Model.hdf5')
        model.save(file_name)
        return model, history
    else:
        return model


def redeNeuralBasica(train_x, validation_x, train_y, validation_y,num_classes,target):

    ############## Rede Neural Funcionando MINHA ####################

    train_x = np.expand_dims(train_x, axis=2)
    validation_x = np.expand_dims(validation_x, axis=2)

    input_shape = train_x.shape[1],1

    #print(input_shape)
    model = models.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=64, activation='elu'))
    model.add(layers.Dense(units=num_classes, activation='softmax'))  # Substitua numero_classes pelo número de classes do problema

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.save("modelAttack.h5")
    history = model.fit(train_x, train_y, epochs=10, batch_size=64, shuffle=True, validation_data=(validation_x, validation_y))

    predictions = model.predict_classes(validation_x)    
    unique_classes = np.unique(validation_y)

    reportMelhor = classification_report(validation_y, predictions,labels=unique_classes, target_names=target)
    print(reportMelhor)

def plotMatrizConf(predictions, validation_y,target, localSalve,experimento):
    
    # Matriz de Confusão
    confusion = confusion_matrix(validation_y, predictions)
    plt.figure(figsize=(9, 5))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
        xticklabels=target, yticklabels=target)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    #plt.savefig(f'{localSalve}/{experimento}-MC.png')
    plt.show()
    plt.clf()

def plotAcuracia(history, localSalve,experimento):

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    plt.legend(loc='lower right')
    #plt.savefig(f'{localSalve}/{experimento}-ACC.png')
    plt.show()
    plt.clf()
    
def plotPRF(report,unique_classes,target,localSalve,experimento):

    unique_classes = [str(class_name) for class_name in unique_classes]

    # Extrair as métricas (precisão, recall e f1-score) para cada classe
    precision = [report[class_name]['precision'] for class_name in unique_classes]
    recall = [report[class_name]['recall'] for class_name in unique_classes]
    f1_score = [report[class_name]['f1-score'] for class_name in unique_classes]

    # Rótulos das classes
    class_labels = target

    # Configurar a largura das barras
    bar_width = 0.2

    # Índice das classes para o eixo x
    index = np.arange(len(class_labels))

    # Criar as barras para as métricas
    plt.bar(index, precision, bar_width, label='Precisão')
    plt.bar(index + bar_width, recall, bar_width, label='Recall')
    plt.bar(index + 2 * bar_width, f1_score, bar_width, label='F1-score')

    # Configurar rótulos do eixo x e legendas
    plt.xlabel('Classes')
    plt.ylabel('Métricas')
    plt.title('Métricas de Classificação por Classe')
    plt.xticks(index + bar_width, class_labels, rotation=45)
    plt.legend()
    #plt.savefig(f'{localSalve}/{experimento}-PRF.png')
    # Exibir o gráfico
    #plt.tight_layout()
    plt.show()