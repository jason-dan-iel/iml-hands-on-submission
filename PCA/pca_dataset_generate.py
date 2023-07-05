# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random

#Generating the dataset
m=100
n=100
dataset=[]
dataset=make_classification(n_samples=m,n_features=n,random_state=40)
dataset

#Implementing PCA from scratch
def PCa(dataset,n_comp):
  dataset=dataset-np.mean(dataset,axis=0)
  cov=np.cov(dataset.T)
  eig_val=np.linalg.eig(cov)[0]
  eigen_vec=np.linalg.eig(cov)[1]
  idx=eig_val.argsort()[::-1]
  eig_val=eig_val[idx]
  eigen_vec=eigen_vec[:,idx]
  proj_mat=eigen_vec[:,:n_comp]
  dataset=np.dot(dataset,proj_mat) 
  return dataset     

reduced_data=PCa(dataset[0],2) 
reduced_data=pd.DataFrame(reduced_data,columns=['PC1','PC2'])  
reduced_data

from sklearn.decomposition import PCA 
pca = PCA(n_components=2) 
x=pca.fit(dataset[0]) 
x1=pca.transform(dataset[0])

reduced_sklearn=pd.DataFrame(x1,columns=['PC1','PC2']) 
reduced_sklearn
