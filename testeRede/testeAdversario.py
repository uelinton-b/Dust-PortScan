import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn import preprocessing
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import save_model, load_model
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import SaliencyMapMethod
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
import tensorflow as tf
import torch.optim as optim
from art.estimators.classification import TensorFlowV2Classifier
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping

tf.compat.v1.disable_eager_execution()
#tf.config.run_functions_eagerly(True)

def treinar_rede(train_x, train_y):

    #train_x, validation_x, train_y, validation_y = train_test_split(X_treino, y_treino, test_size=0.20, shuffle=True, random_state=42, stratify=y)

    # Definir e compilar o modelo da rede neural
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(train_x.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])

    modelo.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    # Treinar o modelo
    modelo.fit(train_x, train_y, epochs=10,batch_size=64,shuffle = True,validation_split=0.1)

    # Salvar o modelo treinado
    modelo.save('IoT_modelo_treinadoBalanceado.h5')
    return modelo


def treinar_redeTiago(train_x, train_y):

    # Splitting the data into train and validation sets if needed
    # train_x, validation_x, train_y, validation_y = train_test_split(X_treino, y_treino, test_size=0.20, shuffle=True, random_state=42, stratify=y)

    # Definindo o callback de Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Monitora a perda de validação
        patience=3,           # Número de épocas sem melhora antes de parar
        mode='min',           # Modo de minimização da perda
        verbose=1             # Mostrar mensagens
    )

    # Defining and compiling the neural network model
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(train_x.shape[1],)),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('tanh'),
        Dense(1,  activation='sigmoid')
    ])

    modelo.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    
    # Training the model
    modelo.fit(train_x, train_y, epochs=150, batch_size=1024, shuffle=True, validation_split=0.1, callbacks=[early_stopping_callback])

    # Saving the trained model
    modelo.save('IoT_modelo_treinadoBalanceadoThiago.h5')
    return modelo

#Scan-1.csv(device_mac,label) + timeseries.csv(device_mac)


arquivo_csv = sys.argv[1]
csv_teste = sys.argv[2]

df1 = pd.read_csv(arquivo_csv, index_col = 0, low_memory=False)
#df1 = pd.read_csv(arquivo_csv, index_col = 0, low_memory=False, nrows = 123658)
df2 = pd.read_csv(csv_teste, index_col = 0, low_memory=False, nrows = 1000000)

#df2 = df2[df2['device_mac'] != 'ff:ff:ff:ff:ff:ff']
df2 = df2.drop(['device_mac','label'],	axis = 1)
#df2 = df2.drop(['device_mac'], axis = 1)
print(df2)

train_x = df1.drop(['device_mac','label', 'label_name'], axis = 1) # seleciona features e ignora a primeira coluna do csv
y = df1['label']  # seleciona rotulos "primeira linha do csv"

#target = list(map(lambda x: 'Normal' if x == 0 else 'Ataque', y))
#target = list(set(target))

le = preprocessing.LabelEncoder()
train_y = le.fit_transform(y)
print(train_y)


oversample = RandomOverSampler(sampling_strategy='not majority', random_state = 42)    
under_sampling = RandomUnderSampler(sampling_strategy='not minority', random_state = 42)

pipeline = Pipeline([
('over_sampling', oversample),
('undersampling', under_sampling)
])

# Ajuste o pipeline aos dados de treinamento

train_again = int(sys.argv[3])

if train_again == 1:
    print("balanceamento dataset........")
    train_x, train_y = pipeline.fit_resample(train_x, train_y)
    modelo = treinar_rede(train_x, train_y)

elif train_again == 2:
    print("balanceamento dataset........")
    train_x, train_y = pipeline.fit_resample(train_x, train_y)
    train_x = train_x.values
    modelo = treinar_redeTiago(train_x, train_y)

else:
    # Carregar o modelo treinado
    modelo = load_model('IoT_modelo_treinadoBalanceado.h5')
    modelito =  load_model('IoT_modelo_treinadoBalanceadoThiago.h5')
    print('\n### -- Modelo carregado -- ###\n')

print("########### -----  Amostras Originais  ----- ###########")
print(df2.values)
print("\n")

model = Sequential([
    Dense(64, activation='relu', input_shape=(train_x.shape[1],)),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('tanh'),
    Dense(1,  activation='sigmoid')
])

model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])


art_classifier = KerasClassifier(model=modelito, use_logits=False) 

attack = FastGradientMethod(estimator=art_classifier, eps=0.5) ## eps = 0.2
print("########### -----  Amostras FGSM  ----- ###########")
x_test_fgsm = attack.generate(x=df2.values)
print(x_test_fgsm)

#attackjsma = SaliencyMapMethod(classifier=art_classifier, theta=0.05, gamma=0.02, batch_size=1,verbose=True)
#print("Starting to Generate JSMA")
#x_test_jsma = attackjsma.generate(x=df2.values)
#print(x_test_jsma)

#print("Starting to Generate PGD")
#attack_pgd = ProjectedGradientDescent(estimator=art_classifier)
#x_test_cw2 = attack_pgd.generate(x=df2.values, verbose = 0)
#print(x_test_cw2)

#y_prob_treino = modelo.predict(train_x)
#y_pred_treino = y_prob_treino.argmax(axis=-1)
#acuracia_treino = accuracy_score(train_y, y_pred_treino)
#print("Acurácia do modelo nos dados de treinamento:", acuracia_treino)
#print('\n')

probabilidades = modelo.predict(x_test_fgsm)
#probabilidades = classifier.predict(df2.values)
#probabilidades = modelo.predict(df2)
print(df2.shape)

# Calcular a estimativa da probabilidade média de ataque
estimativa_probabilidade_ataque = np.mean(probabilidades)

# Exibir a estimativa
print("\nEstimativa da probabilidade média de tráfego estar associado a um ataque:", estimativa_probabilidade_ataque)

limiar = 0.6
predicoes = np.where(probabilidades > limiar,'Ataque','Tráfego Normal')

print("\n")
# Conta o número de instâncias classificadas como ataque e tráfego normal
contagem_ataque = np.sum(predicoes == 'Ataque')
contagem_normal = np.sum(predicoes == 'Tráfego Normal')

# Calcula a proporção de instâncias classificadas como ataque
proporcao_ataque = contagem_ataque / len(predicoes)
proporcao_normal = contagem_normal / len(predicoes)

print("Número de amostras classificadas como ataque:", contagem_ataque)
print("Número de amostras classificadas como tráfego normal:", contagem_normal)
print('\n')
print("[Taxa Verdadeiro positivo] Proporção de amostras classificadas como ataque:", proporcao_ataque)
print("[Taxa Falso Negativo] Proporção de amostras classificadas como tráfego normal:", proporcao_normal)
print("\n")

