#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Task 2
#THE SPARKS FOUNDATION
                                 (GRIPjune21)
Author : SAUMYA Raj Gupta

DATASCIENCE AND BUSINESS ANALYTICS

TASK 2:Prediction using Unsupervised ML

From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually


# In[2]:


#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
from sklearn.cluster import KMeans


# In[4]:


#LOAD DATA
iris = datasets.load_iris()
print(iris)


# In[5]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
#EXPLORATORY DATA ANALYSIS
df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.isna().any


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.corr()


# In[12]:


#GETTING DATA
x= df.iloc[:,:].values
x


# In[13]:


#VISUALISING CLUSTERS
y = iris.target
plt.scatter(x[:,0], x[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Speal Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)


# In[14]:


y = iris.target
plt.scatter(x[:,2], x[:,3], c=y, cmap='gist_rainbow')
plt.xlabel('Petal Length', fontsize=18)
plt.ylabel('Petal Width', fontsize=18)


# In[17]:


#Finding the optimum number of clusters for k-means classification
sos = []
for i in range(1,10):
    km = KMeans(n_clusters = i)
    km.fit(x)
    sos.append(km.inertia_)


# In[18]:


#Plotting the results onto a Elbow graph
plt.plot(range(1,10), sos)
plt.title('Elboe Graph')
plt.xlabel('no.of clusters')
plt.ylabel('sos')
plt.show()


# In[19]:


#Plotting the results onto a Elbow graph
plt.plot(range(1,10), sos)
plt.title('Elboe Graph')
plt.xlabel('no.of clusters')
plt.ylabel('sos')
plt.show()


# In[21]:


#Applying kmeans to the dataset 
Kmeans = KMeans(n_clusters = 3)
y_Kmeans = Kmeans.fit_predict(x)
y_Kmeans


# In[22]:


#Visualising the clusters
plt.scatter(x[y_Kmeans == 0, 0], x[y_Kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_Kmeans == 1, 0], x[y_Kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_Kmeans == 2, 0], x[y_Kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.title('SEPAL LENGTH VS SEPAL WIDTH')
#Plotting the centroids of the clusters
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[23]:


#Visualising the clusters
plt.scatter(x[y_Kmeans == 0, 2], x[y_Kmeans == 0, 3], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_Kmeans == 1, 2], x[y_Kmeans == 1, 3], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_Kmeans == 2, 2], x[y_Kmeans == 2, 3], s = 100, c = 'green', label = 'Iris-virginica')
plt.title('PETAL LENGTH VS PETAL WIDTH')
#Plotting the centroids of the clusters
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




