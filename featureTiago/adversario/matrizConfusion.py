import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_cm(cm,outcome,title,savename):
  print(cm)
  plt.figure(figsize=(20,20),dpi=300)
  plt.matshow(cm, cmap="OrRd" )
  plt.title(title, pad=100)
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  #plt.ylabel('Rótulo verdadeiro')
  #plt.xlabel('Rótulo classificado')
  ax = plt.gca()
  from matplotlib.ticker import MultipleLocator; ax.xaxis.set_major_locator(MultipleLocator(1)); ax.yaxis.set_major_locator(MultipleLocator(1))
  l_col_list = list(outcome)
  import string
  for i in range(0,len(l_col_list)):
    pretty_string = ''.join(filter(lambda x: x in string.printable, l_col_list[i]))
    l_col_list[i] = pretty_string

  ax.set_xticklabels([''] + l_col_list, rotation=90)
  ax.set_yticklabels([''] + l_col_list)

  # savename='ConfusionMatrix-Multiclass-Original.png'
  basedir='Figura/'
  filename=basedir+savename
  plt.savefig(filename,dpi=300,bbox_inches='tight')
  plt.show()

def plot_errors(cm,outcome,title,savename):
  #Plot of errors
  row_sums = cm.sum(axis=1, keepdims=True)
  norm_cm = cm / row_sums

  np.fill_diagonal(norm_cm, 0)
  plt.figure(figsize=(20,20),dpi=300)
  plt.matshow(norm_cm, cmap="OrRd")
  plt.title(title, pad=100)

  plt.colorbar()
  ax = plt.gca()
  from matplotlib.ticker import MultipleLocator; ax.xaxis.set_major_locator(MultipleLocator(1)); ax.yaxis.set_major_locator(MultipleLocator(1))
  l_col_list = list(outcome)
  import string
  for i in range(0,len(l_col_list)):
    pretty_string = ''.join(filter(lambda x: x in string.printable, l_col_list[i]))
    l_col_list[i] = pretty_string

  ax.set_xticklabels([''] + l_col_list,rotation=90)
  ax.set_yticklabels([''] + l_col_list)

  # savename='ErrorMatrix-Multiclass-Original.png'
  basedir='Figura/'
  filename=basedir+savename
  plt.savefig(filename,dpi=300,bbox_inches='tight')



def matrizdeconfusao(rounded_cw2_predictions, outcome,y_test):
    

  cm = confusion_matrix(y_test,rounded_cw2_predictions)

  plot_cm(cm,outcome,'Matriz de Confusão','matriz.png')
  plot_errors(cm,outcome,'Matriz de Confusão de Erro','erroMatriz.png')


#outcome = np.load(path + 'outcome.npy', allow_pickle=True) ## label