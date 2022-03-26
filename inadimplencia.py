#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar os pacotes necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, plot_confusion_matrix, accuracy_score, f1_score

# filtrar mensagens de warning
import warnings
warnings.filterwarnings('ignore')


# ##### Importando os dados

# In[2]:


df = pd.read_csv("inadimplencia.csv", na_values= 'na')


# ###### Análise Exploratória

# In[3]:


###### Dimensão e proporção dos dados
df.total = df.shape

print(f'total: {df.total}')

df.N_inad = df[df['default'] == 0].shape

df.inad = df[df['default'] == 1].shape

print(f'inadimplentes: {round(100 * df.inad[0] / df.total[0], 2)}')

print(f'não inadimplentes: {round(100 * df.N_inad[0] / df.total[0], 2)}')


# In[4]:


### tipos das variavéis
df.dtypes


# In[5]:


### estatística descritiva
df.drop('id', axis=1).select_dtypes('number').describe().transpose()


# In[6]:


#verificando se contém valores faltantes
df.isna().any()


# In[7]:


### removendo valores faltantes 
df.dropna(inplace = True)


# In[8]:


### transformando variáveis categóricas em numéricas
df['sexo'] = df['sexo'].replace('M', 1).replace('F', 0)

df['limite_credito'] = df['limite_credito'].apply(lambda valor: float(valor.replace(".", "").replace(",", ".")))

df.head()


# In[9]:


import plotly.express as px
graph = px.treemap(df, path = ['tipo_cartao', 'idade'])


# In[10]:


#### vericando se os dados estão desbalanceados
sns.countplot(x = df['default']);


# ##### Machine Learning

# In[11]:


#labelencorder = transformar variavéis categóricas em numericas
from sklearn.preprocessing import LabelEncoder

#instânciando o objeto
LEt = LabelEncoder()

df['salario_anual'] = LEt.fit_transform(df['salario_anual'])


# In[12]:


# separando as variáveis dependentes e independente
X = df.iloc[:, [2, 3, 7, 13]]
y = df.iloc[:, 1]

# separando em treino e teste
validation_size = 0.20
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state = seed)


# In[13]:


# normalizando
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[14]:


# importação do algoritmo de ML
from sklearn.neighbors import KNeighborsClassifier


# In[15]:


# definindo o modelo
knn = KNeighborsClassifier(n_neighbors=13, metric='euclidean')
knn.fit(X_train, y_train)


# In[16]:


#Previsões
y_previsto = knn.predict(X_test)
y_previsto


# ##### Avaliando o Modelo

# In[17]:


from sklearn import metrics


# In[18]:


matriz_de_confusao = confusion_matrix(y_test,y_previsto)
print(matriz_de_confusao)


# In[19]:


# essa métrica considera tanto o recall como a precisão
f1_score(y_test, y_previsto)


# In[20]:


# acerto bruto
accuracy_score(y_test, y_previsto)


# In[24]:


# precisão
result = knn.predict(X_test)
precision = precision_score(y_test, result)

