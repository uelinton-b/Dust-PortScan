    # %%
import json
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn import preprocessing
import tensorflow as tf
from sklearn.cluster import KMeans
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform, glorot_normal, he_normal
from keras.optimizers import Adamax, RMSprop, SGD, Adadelta
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sys
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


#load_path = "Trained_Model_ALLRotulo.hdf5"
#load_path = "Trained_Model_1Rotulo.hdf5"
load_path = "Trained_Model_normal_and_Portbinario.hdf5"
model = load_model(load_path)


csv_arquivo = sys.argv[1]

#df1 = pd.read_csv(arquivo_csv, index_col = 0, low_memory=False, nrows=1000000)

df2 = pd.read_csv(csv_arquivo, index_col = 0, low_memory=False)

#df2 = df2[df2['device_mac'] != 'ff:ff:ff:ff:ff:ff']
#df2 = df2.drop(['device_mac'], axis = 1)
print(df2)


#df2 = df2[(df2['device_name'] == 'Assistant1')
#| (df2['device_name'] == 'Wswitch1') | (df2['device_name'] == 'Laptop2')] 

#df2 = df2.drop(columns=['device_mac','device_name'])

#X = df1.drop(['device_mac'], axis = 1) # seleciona features e ignora a primeira coluna do csv
#y_string = df1['device_mac']  # seleciona rotulos "primeira linha do csv"

#train = np.expand_dims(X, axis=2)
df2NP = np.expand_dims(df2.values, axis=2)
    
#predictions = model.predict(df2NP)
#predict_classes

#print(predictions)

df2NP = df2NP/255

ynew = model.predict(df2NP)
# show the inputs and predicted outputs
#for i in range(len(df2NP)):
 #print("X=%s, Predicted=%s" % (df2NP[i], ynew[i]))

print(ynew)
print(ynew[0])
print([1])

estimativa_probabilidade_ataque = np.mean(ynew)

# Exibir a estimativa
print("Estimativa da probabilidade média de tráfego estar associado a um ataque:", estimativa_probabilidade_ataque)
exit()

print(ynew)

# Define um limiar de probabilidade
limiar = 5.0

for probabilidade in ynew:
    
    print(probabilidade[1])

    if probabilidade[0] > limiar:
        print('Maior', probabilidade[0])
    else:
        print('limiar menor', probabilidade[0])
    print('\n')
    time.sleep(3)

# Classifica as instâncias com base no limiar
predicoes = np.where(ynew > limiar,'Ataque','Tráfego Normal')

# Conta o número de instâncias classificadas como ataque e tráfego normal
contagem_ataque = np.sum(predicoes == 'Ataque')
contagem_normal = np.sum(predicoes == 'Tráfego Normal')

# Calcula a proporção de instâncias classificadas como ataque
proporcao_ataque = contagem_ataque / len(predicoes)
proporcao_normal = contagem_normal / len(predicoes)

print("Número de instâncias classificadas como ataque:", contagem_ataque)
print("Número de instâncias classificadas como tráfego normal:", contagem_normal)
print("Proporção de instâncias classificadas como ataque:", proporcao_ataque)
print("Proporção de instâncias classificadas como tráfego normal:", proporcao_normal)

exit()
'''
# 1. Análise de Componentes Principais (PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df2NP.reshape(df2NP.shape[0], -1))

# 2. Visualização de Clusters (t-SNE)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(df2NP.reshape(df2NP.shape[0], -1))

# Plot PCA result
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], cmap='viridis')
plt.title('PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Probabilidade')
plt.show()

# Plot t-SNE result
plt.figure(figsize=(10, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], cmap='viridis')
plt.title('t-SNE')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Probabilidade')
plt.show()

# 3. Análise de Densidade
sns.kdeplot(tsne_result[:, 0], tsne_result[:, 1], cmap='viridis', shade=True, shade_lowest=False)
plt.title('Densidade de t-SNE')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
'''


classe_predita = np.argmax(ynew)

# Exibir a classe predita
print("Classe Predita:", classe_predita)

predictions = ynew

plt.hist(predictions.max(axis=1), bins=50, edgecolor='black')
plt.title('Confiança das Previsões')
plt.xlabel('Confiança')
plt.ylabel('Frequência')
plt.show()

indices_a_visualizar = [0, 1]
for i in indices_a_visualizar:
    plt.plot(predictions[i], label=f'Exemplo {i + 1}')

plt.title('Exemplos de Previsões')
plt.xlabel('Classes')
plt.ylabel('Probabilidade')
plt.legend()
plt.show()

threshold = 0.6  # Define um valor de limite
binary_predictions = (predictions > threshold).astype(int)

count_0 = 0
count_1 = 0
#count_2 = 0

# Visualizar as previsões e rótulos
for i in range(len(binary_predictions)):
    #print(f"Amostra {i + 1}: Previsão = {binary_predictions[i][0]}")
    if binary_predictions[i][0] == 0:
        count_0 += 1
    elif binary_predictions[i][0] == 1:
        count_1 += 1
    #elif binary_predictions[i][0] != 1 and binary_predictions[i][0] !=0:
    #    count_2 += 1

# Imprimir os resultados
print(f"Total de previsões 0: {count_0}")
print(f"Total de previsões 1: {count_1}")
#print(f"Total de previsões 2: {count_2}")

plt.plot(predictions)
plt.xlabel('Exemplos')
plt.ylabel('Previsões')
plt.title('Previsões do Modelo')
plt.show()

# Lista para armazenar as previsões do modelo
predictions_list = []

# Calcule a classe prevista com base nas probabilidades
top_k = 3  # Número de principais rótulos para exibir
top_indices = np.argsort(predictions[0])[::-1][:top_k]  # Índices dos principais rótulos
top_labels = [str(i) for i in top_indices]  # Converte os índices em strings
top_probabilities = predictions[0][top_indices]  # Probabilidades dos principais rótulos

print("\n2 principais rótulos previstos:")
for label, probability in zip(top_labels, top_probabilities):
    print(f"Classe {label}: Probabilidade {probability:.4f}")

print('\n')
# Função para calcular a confiança do modelo para uma amostra de dados
def calcular_confianca(amostra):
    previsoes = model.predict(amostra)[0]  # Faz previsões para a amostra
    classe_predita = np.argmax(previsoes)
    probabilidade_maxima = np.max(previsoes)
    entropia = -np.sum(previsoes * np.log(previsoes))  # Calcula a entropia da distribuição de probabilidades
    margem = np.max(previsoes) - np.sort(previsoes)[-2]  # Calcula a diferença entre a maior e a segunda maior probabilidade
    #margem = 1

    return classe_predita, probabilidade_maxima, entropia, margem

# Exemplo de cálculo de confiança para uma amostra de dados
amostra = df2NP  # Substitua com os valores reais da sua amostra de dados
classe_predita, probabilidade_maxima, entropia, margem = calcular_confianca(amostra)
print("Classe prevista:", classe_predita)
print("Probabilidade máxima:", probabilidade_maxima)
print("Entropia:", entropia)
print("Margem de classificação:", margem)
print('\n')




'''
# Exemplo de k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(predictions)

for cluster_num in range(max(cluster_labels) + 1):
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_num]
    
    print(f"Exemplos no Cluster {cluster_num}:")
    for idx in cluster_indices[:min(5, len(cluster_indices))]:  # Imprima no máximo 5 exemplos por cluster
        print(f"Exemplo {idx + 1}: {predictions[idx]}")
    
    print("\n")
'''