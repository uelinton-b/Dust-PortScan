import json
import time
import pandas as pd
import numpy as np
import random
import sys
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from startFCC import *
import time
from time import sleep
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from collections import Counter 

def main():

    if len(sys.argv) != 3:
        print("Uso: python3 script.py arquivo.csv experimento")
        return

    arquivo_csv = sys.argv[1]
    experimento = sys.argv[2]

    # Medição do tempo de execução
    start_time = time.time()

    #df1 = pd.read_csv(arquivo_csv, low_memory=False)
    df1 = pd.read_csv(arquivo_csv, index_col = 0, low_memory=False)

    #df1 = df1[~df1['device_name'].str.startswith('Fake')]
    #df1 = df1[~df1['device_name'].astype(str).str.startswith('Fake')]

    #df1 = df1[(df1['device_name'] == 'BabyMonitor1') | (df1['device_name'] == 'Camera1') | (df1['device_name'] == 'Device1')] 

    #df1 = df1[df1['device_name'] != "Gateway"]
    #(4304330, 21)
    print(df1.shape)

    print(df1)
    print(df1.columns)

    #contagem = df1['label'].value_counts()
    #print(contagem)

    #df1 = df1[(df1['device_name'] == 'Sleep1') | (df1['device_name'] == 'Assistant1') | (df1['device_name'] == 'Camera5')]

    #X = df1.drop(['device_mac','device_name'], axis = 1) # seleciona features e ignora a primeira coluna do csv
    #y_string = df1['device_name']  #seleciona rotulos "primeira linha do csv"

    #X = df1.drop(['device_mac'], axis = 1) # seleciona features e ignora a primeira coluna do csv
    #y_string = df1['device_mac']  #seleciona rotulos "primeira linha do csv"

    #print(df1.columns)

    #X = df1.drop(['Attack_type','Attack_label'], axis = 1) # seleciona features e ignora a primeira coluna do csv
    #y_string = df1['Attack_type']  # seleciona rotulos "primeira linha do csv"

    X = df1.drop(['device_mac','label'], axis = 1) # seleciona features e ignora a primeira coluna do csv
    y_string = df1['label']  # seleciona rotulos "primeira linha do csv"

    #X = df1.drop(['device_name','day_index', 'day_capture'], axis = 1) # seleciona features e ignora a primeira coluna do csv
    #y_string = df1['device_name']  #seleciona rotulos "primeira linha do csv"

    #X.to_csv("train_xPortScanningAttack_and_Modbus.csv")
    #exit()

    quantRotulos = len(y_string.unique())
    #target = list(map(lambda x: 'Normal' if x == 0 else 'Ataque', y_string))
    #target = list(set(target))
    target = y_string.unique()
    print(target)

    ## transformando label em inteiro
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y_string)
    

    #X, y = janelaDeslisante(X, y) 

    #oversample = BorderlineSMOTE()
    #oversample = SMOTE(random_state=42)
    #oversample = SVMSMOTE()

    print("balanceamento dataset........")
    oversample = RandomOverSampler(sampling_strategy='not majority', random_state = 42)    
    under_sampling = RandomUnderSampler(sampling_strategy='not minority', random_state = 42)
    
    #X, y = under_sampling.fit_resample(X, y)

    #X, y = oversample.fit_resample(X, y)

    #X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Divisão do restante em treino e validação
    #X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    #redeNeuraldosGuri(X_train, y_train, X_val, y_val, X_test, y_test, quantRotulos,target)


    pipeline = Pipeline([
    ('oversampling', oversample),
    ('undersampling', under_sampling)
    ])

    # Ajuste o pipeline aos dados de treinamento
    X, y = pipeline.fit_resample(X, y)

    #plot_feature_proportions(X, y, target)

    #plotProporcaoFeatures(quantRotulos, target, X, y)

    #X.to_csv("train_Normal.csv")
    #exit()

    train_x, validation_x, train_y, validation_y = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42, stratify=y)

    #print(validation_x)
    #validation_x.to_csv("train_x.csv")
    #exit()
    unique_classes = np.unique(y)

    weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    class_weights = {class_label: weight for class_label, weight in zip(le.classes_, weights)}
    
    for class_label in range(quantRotulos):
        if class_label not in class_weights:
            class_weights[class_label] = 1.0
    
    #redeNeuralBasica(train_x, validation_x, train_y, validation_y, quantRotulos,target)
    redestruturada(train_x, train_y, validation_x, validation_y, quantRotulos, target, experimento, class_weights)


    end_time = time.time()  # Marca o tempo final
    # Calcula o tempo total de execução
    temp = end_time - start_time
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    print('Tempo de Execução:','%d:%d:%d' %(hours,minutes,seconds))


def redestruturada(train_x, train_y, validation_x, validation_y, NUM_OF_CLASSES, target, experimento, class_weights):
    
    BATCH_SIZE = 64
    Train_classifier = 1
    EPOCH = 10
    localSalve = "resultados/"

    nameModeloSalve = "Trained_Model_normal_and_PortBinario.hdf5"

    treinar = 1

    with open('models.json') as json_file:
        models = json.load(json_file)
    for model_index in range(len(models)):
        print("model", model_index)

        if Train_classifier == 1:
            model_path = None
        else:
          if FCC == "FCC-HP":
              model_path = "Trained_ModelportScan.hdf5"
          if FCC == "FCC-P":
              model_path = "Trained_ModelportScan.hdf5"

        if treinar == 1:
        
            flow_classification_model, history = train(treinar, train_x/255, train_y, validation_x/255,validation_y,EPOCH, class_weights,  
                                                          model_info=models[model_index],file_name=nameModeloSalve,
                                                          num_classes=NUM_OF_CLASSES,batch_size=BATCH_SIZE, load_path= model_path) 
        
            expanded_test_x = np.expand_dims(np.array(validation_x / 255), axis=2)
            #pred = flow_classification_model.predict(expanded_test_x)
            pred = flow_classification_model.predict_classes(expanded_test_x)
            unique_classes = np.unique(validation_y)

            reportMelhor = classification_report(validation_y, pred,labels=unique_classes, target_names=target)
            print(reportMelhor)

            plotAcuracia(history, localSalve, experimento)
            plotMatrizConf(validation_y, pred, target, localSalve, experimento)
            
            report = classification_report(validation_y, pred, labels=unique_classes, output_dict=True)
            plotPRF(report,unique_classes,target,localSalve, experimento)
        
        else:
            modelo = train(treinar, train_x/255, train_y, validation_x/255,validation_y,EPOCH, class_weights,  
                                                          model_info=models[model_index],file_name=nameModeloSalve,
                                                          num_classes=NUM_OF_CLASSES,batch_size=BATCH_SIZE, load_path= model_path) 

            fold(train_x, train_y, validation_x, validation_y, NUM_OF_CLASSES, modelo)
        

def fold(train_x, train_y, validation_x, validation_y, NUM_OF_CLASSES, modelo):

    # Definindo o número de folds para validação cruzada
    num_folds = 5
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Lista para armazenar os resultados de cada fold
    resultados = []
    train_x = train_x.values
    # Loop sobre os folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        print(f"Treinando no fold {fold + 1}...")
        # Extraia conjuntos de treinamento e validação para este fold
        fold_train_x, fold_val_x = train_x[train_idx], train_x[val_idx]
        fold_train_y, fold_val_y = train_y[train_idx], train_y[val_idx]

        fold_train_x = np.expand_dims(np.array(fold_train_x), axis=2)
        fold_val_x = np.expand_dims(np.array(fold_val_x), axis=2)

        fold_train_y = to_categorical(fold_train_y, num_classes=NUM_OF_CLASSES)
        fold_val_y = to_categorical(fold_val_y, num_classes=NUM_OF_CLASSES)

        # Treine o modelo
        modelo.fit(fold_train_x, fold_train_y, epochs=10, batch_size=64, validation_data=(fold_val_x, fold_val_y))

        # Avalie o modelo no conjunto de validação
        resultado = modelo.evaluate(fold_val_x, fold_val_y)
        resultados.append(resultado)

    # Calcule a média e o desvio padrão dos resultados
    resultados = np.array(resultados)
    media_acuracia = np.mean(resultados[:, 1])
    desvio_padrao_acuracia = np.std(resultados[:, 1])

    # Calcule a média e o desvio padrão dos resultados
    resultados = np.array(resultados)
    media_acuracia = np.mean(resultados[:, 1])
    desvio_padrao_acuracia = np.std(resultados[:, 1])

    print(f"Acurácia média: {media_acuracia:.4f} ± {desvio_padrao_acuracia:.4f}")

    # Avalie o modelo no conjunto de teste
    acuracia_teste = modelo.evaluate(fold_val_x, fold_val_y)[1]
    print(f"Acurácia no conjunto de teste: {acuracia_teste:.4f}")


def janelaDeslisante(data, labels):
    window_size = 3
    
    windows = []
    sample_labels = []

    #print(data)
    #print(labels)

    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i + window_size]
        #print(window)

        label = labels[i + window_size - 1]
        #print(label)

        windows.append(window.values.flatten())
        sample_labels.append(label)

    windows = np.array(windows)
    sample_labels = np.array(sample_labels)

    return windows, sample_labels

def plotProporcaoFeatures(quantRotulos, target, X, y):

    print("Proporcao por Rotulo...............")

    # Criando uma paleta de cores com base no número de rótulos
    palette = sns.color_palette("husl", quantRotulos)

    for i in range(quantRotulos):
        samples_ix = np.where(y == i)[0]
        plt.scatter(X.iloc[samples_ix, 0], X.iloc[samples_ix, 1], label=f'{target[i]}', color=palette[i])

    # Adicionar legenda
    plt.legend(title='Rótulos')

    # Mostrar o gráfico
    plt.show()

def plot_feature_proportions(X, y, label_names):
    
    print("Proporcao por Rotulo...............")
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Distribuição dos Dados')
    plt.show()

    # Análise da distribuição das classes
    counter = Counter(y)
    print("Distribuição das Classes:", counter)




if __name__ == "__main__":
    main()




# Gateway         2560510
# Camera4          676719
# Camera3          185446
# Laptop1          175791
# Camera5          104731
# Assistant1        97121
# Wswitch1          84892
# Camera1           70512
# Device1           64081
# Camera2           48571
# Tablet1           46355
# Motion1           43466
# Sleep1            31672
# Laptop2           25783
# Camera6           22374
# LightBulb1        16677
# Printer1          10501
# Speaker1           8000
# Weather1           6460
# BabyMonitor1       5159
# Picturef1          4570
# Phone2             4510
# iHome1             3824
# Splug1             3195
# Laptop3            1156
# Phone1              804
# Scale1              694
# Smoke1              465
# Phone3              170
# Device2              87
# BlipcareBP1          34
