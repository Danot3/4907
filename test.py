# import required libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Import dataset with pandas
#df = pd.read_csv ("114_congress.csv")
'''
test_data = np.zeros((40000, 4))
test_data[0:10000, :] = 30.0
test_data[10000:20000, :] = 60.0
test_data[20000:30000, :] = 90.0
test_data[30000:, :] = 120.0
df = test_data

# Remove the label column 
df = df.drop(['name','party','state'], axis=1)
df.info()
'''
# Import dataset with pandas
df = pd.read_csv ("set_D2.csv")

# Remove the label column 
df = df.drop(['Unnamed: 0'], axis=1)
df.info()

# Standardize the data features onto unit scale 
x = np.array(df)
scaler = StandardScaler()
# verify scale values 
print(scaler.fit(x))
print(scaler.mean_)
x = scaler.transform(x)
print(x)

# Apply PCA decomposition to reduce dimensions to 2
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
# Set the decomposed as a data frame
principalDF = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
# Convert back into array 
principals = np.array(principalDF)
print(principalDF)

# Create graph output for PCA 
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20) 
ax.scatter(principals[:,0], principals[:,1])
ax.grid()


kmeans = KMeans(n_clusters=4,random_state=0).fit(principals)
bx = fig.add_subplot(1,1,1)
bx.set_xlabel('Principal Component 1', fontsize = 15)
bx.set_ylabel('Principal Component 2', fontsize = 15)
bx.set_title('2 component PCA', fontsize = 20) 
bx.scatter(principals[:,0], principals[:,1])
bx.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:, 1], s=300, c='red')
bx.grid()
plt.show()


"""
#elbow method
plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('dataset')
plt.scatter(principals[:,0],principals[:,1])
plt.show()

x1 = principals[:,0]
x2 = principals[:,1]

plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b’, ‘g’, ‘r']
markers = ['o’, ‘v’, ‘s']


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""


#silhouette score 
range_n_clusters = list (range(2,10))
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(principals)
    centers = clusterer.cluster_centers_

    score = silhouette_score(principals, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))





