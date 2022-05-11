# Import Statements
from calendar import c
from turtle import color
import pandas as pd
import numpy as np
from pyparsing import col
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



# Implementazione Algoritmo PCA
def PCA(scaled_data, n=2):
    
    # Calcoliamo la matrice delle covarianze
    matrice_covarianze = np.cov(scaled_data.T)
    
    # Calcoliamo autovalori e autovettori della matrice delle covarianze
    autovalori, autovettori = np.linalg.eig(matrice_covarianze)
    
    # Ordiniamo gli autovettori in base ai corrispondenti autovalori
    autovettori = autovettori.T
    indice = np.argsort(autovalori)[::-1]
    
    autovalori, autovettori = autovalori[indice], autovettori[indice]
    
    # Ottengo i primi n autovettori
    vettori_pca = autovettori[0:n]
    
    # Prinicipal Components
    valori_pca = np.dot(scaled_data, vettori_pca.T)
    
    return valori_pca



# Load the Sample Dataset
df = pd.read_csv('iris.csv'
                 , names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values

y = df.loc[:,['target']].values

sample_df = pd.DataFrame(data=x,columns=features)
  

#Performing Standardization
standard_df = (sample_df - sample_df.mean()) / sample_df.std()
standard_df.head(10)

# Applying PCA to the dataset
principal_comps = PCA(standard_df, n=2)

principalDf = pd.DataFrame(data = principal_comps
             , columns = ['principal component 1', 'principal component 2'])

principalDf.head(5)

df[['target']].head()

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(5)


##########
figu = plt.figure(figsize = (8,8))
ax1 = figu.add_subplot(1,1,1) 
ax1.set_xlabel('Principal Component 1', fontsize = 15)
ax1.set_ylabel('Principal Component 2', fontsize = 15)
ax1.set_title('2 Component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colori = ['r', 'g', 'b']
for target, color in zip(targets,colori):
    indicesToKeep = finalDf['target'] == target
    ax1.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax1.legend(targets)
ax1.grid()

plt.show()
