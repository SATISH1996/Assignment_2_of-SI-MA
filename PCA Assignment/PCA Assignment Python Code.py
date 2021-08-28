#!/usr/bin/env python
# coding: utf-8

# ### Loading Dataset and importing required packages

# In[1]:


from sklearn.cluster import KMeans # Our clustering algorithm
from sklearn.decomposition import PCA # Needed for dimension reduction
from sklearn.datasets import load_wine # Dataset that I will be using
import matplotlib.pyplot as plt # Plotting 
import pandas as pd # Storing data convenieniently
import numpy as np # Needed for numerical calculations


# In[2]:


wines = load_wine()
wine_df = pd.DataFrame(wines.data, columns=wines.feature_names)
wine_df.head()


# In[3]:


wine_df.shape


# In[4]:


wine_df.info()


# In[5]:


wine_df.describe()


# ### (A). PCA to perform dimension reduction

# In[6]:


# normalizing the data 
from sklearn.preprocessing import StandardScaler
std_wine = StandardScaler().fit_transform(wine_df)


# In[7]:


pca = PCA(n_components=13)
principalComponents = pca.fit_transform(std_wine)


# In[8]:


# Plotting the variances for each PC
PC = range(1, pca.n_components_+1)
plt.bar(PC, pca.explained_variance_ratio_, color='orange')
plt.xlabel('Principal Components')
plt.ylabel('Variance %')
plt.xticks(PC)
plt.show()


# In[9]:


# Putting components in a dataframe 
PCA_components = pd.DataFrame(principalComponents)
PCA_components.head()


# In[10]:


PCA_components.shape


# In[11]:


#Variation explained by top 5 Principal Components
np.sum(pca.explained_variance_ratio_[:5])


# In[12]:


# Most of the variablity in data is explained by top 5 PC's so we can reduce the dimension to 5 from 13 features originally
Reduced_df= PCA_components.iloc[:, :5]


# In[13]:


Reduced_df.shape


# ### (B1). Scatter plot of first two Principal Components

# In[14]:


plt.scatter(PCA_components[0], PCA_components[1], alpha=.3, color='blue')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# ### Cluster Analysis

# In[15]:


#Deciding Number of clusters using Elbow Chart
inertias = []

# Creating 10 K-Mean models while varying the number of clusters (k)
for k in range(1,10):
    model = KMeans(n_clusters=k)
    
    # Fit model
    model.fit(Reduced_df)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(range(1,10), inertias, '-p', color='green')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
ks=range(1,10)
plt.xticks(ks)
plt.show()


# In[16]:


# Elbow point is at three number of clusters. So we will make 3 clusters


# In[17]:


model = KMeans(n_clusters=3)
model.fit(Reduced_df)

labels = model.predict(Reduced_df)
plt.scatter(Reduced_df[0], Reduced_df[1], c=labels, edgecolors ='black')
plt.show()


# ### (B2). Linear Regression Analyis

# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[19]:


#Choosing First PC as Target, rest as features
X = Reduced_df.iloc[:, [1,2,3,4]]
y = Reduced_df.iloc[:, [0]]
print(X.shape, y.shape)


# In[20]:


# Train Test Split - 70% Train / 30% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


# In[21]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[22]:


r_sq = model.score(X_train, y_train)
print('coefficient of determination for Train Set:', r_sq)


# In[23]:


r_sq_test = model.score(X_test, y_test)
print('coefficient of determination for Test Set:', r_sq_test)


# In[24]:


print('intercept:', model.intercept_)
print('Features regression Coeff:', model.coef_)


# In[ ]:




