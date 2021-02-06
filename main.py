#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import required libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Import dataset with pandas
df = pd.read_csv ("set_D2.csv")


# In[2]:


# remove the label 
df = df.drop(['Unnamed: 0'], axis=1)
df.head()


# In[9]:


# standardize the data features onto unit scale 
x = np.array(df)
scaler = StandardScaler()
scaler.fit(x)
scaler.mean_
x = scaler.transform(x)
# add PCA decomposition to 2 variables 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principals = np.array(principalDF)
principalDF.head()

# In[10]:
    #
def revertData(numberPoints, values, groupCenter, groupCluster):
    
    centerPoints = []
    for centers in groupCenter:
        i = 0
        currentClosest = []
        #print('a')
        while i < numberPoints:
            currentClosest.append(-1)
            i+=1
            #print('b')
        centerPoints.append(currentClosest)
    #print(centerPoints)        
    #print('test center', groupCenter)
    for j,point in enumerate(values):
        #print('point ', j, 'is ', point, 'in group ', groupCluster[j])
        
        
        group = groupCluster[j]
        i=0
        
        while i < len(centerPoints[group]):
            #print(centerPoints[group])
            #print('i test ', centerPoints[group][i])
            centerPoint = groupCenter[group]
            newX = abs(abs(centerPoint[0])-abs(point[0]))
            newY = abs(abs(centerPoint[1])-abs(point[1]))
            newAvg = (newX + newY)/2
            
            closestPoints = centerPoints[group][i]
            #print('current closest ', closestPoints, 'against ', point)
            
            if closestPoints == -1:
                oXTemp = (10.0)
                oYTemp = (10.0)
            else:
                oXTemp = (values[closestPoints][0])
                oYTemp = (values[closestPoints][1])
            
            oldX = abs(abs(centerPoint[0])-abs(oXTemp))
            oldY = abs(abs(centerPoint[1])-abs(oYTemp))
            oldAvg = (oldX + oldY)/2
            #print('new avg= ', newAvg, ' old avg= ', oldAvg)
            if newAvg < oldAvg:
                #print(i, ' ', centerPoints[i], ' ', point)
                centerPoints[group][i] = j
                #print('Updated')
                break
            i+=1
            
    #print('done ', centerPoints)
    return(centerPoints)
        
 # In[11]:     
def centerValue(df, closestValues):
    originalValues = df.values.tolist()
    #print(originalValues[closestValues[0][0]])
    centerValues = []
    for cluster in closestValues:
        #print(cluster)
        i = 0
        dataPoints = []
        while i < len(originalValues[0]):
            tempValue = 0
            for point in cluster:
                tempValue += originalValues[point][i]
            dataPoints.append(tempValue/len(cluster))
            i+=1
        #print(dataPoints)
        centerValues.append(dataPoints)
    return(centerValues)
     
# In[31]:


# use numpy array to convert to tensor object
def input_fn():
    return tf.train.limit_epochs(
        tf.convert_to_tensor(principals, dtype=tf.float32), num_epochs=1)


# In[37]:


# define cluster amount and setup kmeans 
num_clusters = 2
kmeans = tf.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False)


# In[38]:


# train
num_iterations=10
previous_centers=None
for _ in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
        print ('delta: ', cluster_centers - previous_centers)
    previous_centers = cluster_centers
    print ('score: ', kmeans.score(input_fn))
print ('cluster centers: ', cluster_centers)


# In[42]:


# list point to centroid 
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(principals):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)

nearPoints = 3
weightedPoints = revertData(nearPoints, principals, cluster_centers, cluster_indices)
for i,points in enumerate(weightedPoints):
    print('Closest ', nearPoints, ' points to center ', i, 'are ', points)
    
centerValues = centerValue(df,weightedPoints)
print(centerValues)

# In[51]:


# output graph 
plt.scatter(principals[:,0],principals[:,1], c=cluster_indices, cmap='rainbow')
plt.scatter(cluster_centers[:,0] ,cluster_centers[:,1], color='black')



# In[]:
    
