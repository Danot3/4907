# import required libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_samples, silhouette_score
from math import sqrt # For standardization 
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# for 3d plot
from mpl_toolkits import mplot3d
from tkinter import *

def readInput():
    global df_data
    global input_text
    df_location = input_text.get("1.0",'end-1c')
    # Import dataset with pandas
    df = pd.read_csv (df_location)
    df_removed_header = df 
    # Remove the first column which should be student numbers (or any other index scheme)
    # Obtain list of columns indices 
    df_column_numbers = [x for x in range(df.shape[1])]
    # Remove the column at index 0
    df_column_numbers.remove(0)
    # Return original data without the first column 
    df_removed_header.iloc[:, df_column_numbers] 
    # Check against any missing data within the file and replace with the mean of the feature
    df_filled_data = df_removed_header.fillna(df_removed_header.mean())
    df_data = df_filled_data

# Manual standardization           
# calc column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means
 
# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs
 
# standardize dataset by applying formula to each value in the dataset 
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
            

def standardize():
    global standardizedData
    global df_data
    dset = np.array(df_data)
    dset1 = dset.astype(np.float)
    means = column_means(dset1)
    stdevs = column_stdevs(dset1, means)
    standardize_dataset(dset1, means, stdevs)
    standardizedData = dset1

def PCA():
    global standardizedData
    global pcaData
    standardize() 
    M = mean(standardizedData.T, axis=1) #Calculate mean of the dataset
    C = standardizedData - M # Center columns by subtracting the mean from the dataset
    V = cov(C.T) #Find the covariance matrix from the centered dataset
    values, vectors = eig(V) #Find the eigenvalues and eigenvectors of the covariance matrix
    P = vectors.T.dot(C.T) #Project the data using the vectors 
    P = P.T * -1 # *-1 the data to match scikit 
    principalDF = P[:, :2] # Drop all dimensions except the first two (ones we want for 2d)
    principalDF = pd.DataFrame(data = principalDF #Convert to pandas dataframe 
                 , columns = ['principal component 1', 'principal component 2'])
    P = P[:, :2]
    pcaData = P
    return P

def plotPCA():
    global pcaData
    # Plot a graph of the data after PCA
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20) 
    ax.scatter(pcaData[:,0], pcaData[:,1])
    fig.savefig('Output/PCA_Decomposed_Graph.png')
    plt.close(fig)

def elbowGraph():
    global pcaData
    # Output cluster to distance graph to identify optimal cluster amount
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(pcaData)
        Sum_of_squared_distances.append(km.inertia_)
    fig = plt.figure(figsize = (10,10))
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    fig.savefig('Output/Elbow_Graph.png')
    plt.close(fig) 
    
# use numpy array to convert to tensor object
def input_fn():
    global pcaData
    return tf.train.limit_epochs(
        tf.convert_to_tensor(pcaData, dtype=tf.float32), num_epochs=1)

def kmeansTF():
    # define cluster amount and setup kmeans 
    global num_clusters 
    global kmeans
    global clusters_text
    num_clusters = clusters_text.get("1.0",'end-1c')
    if num_clusters.isdigit():
        num_clusters = int(num_clusters)
        kmeans = tf.estimator.experimental.KMeans(
            num_clusters=num_clusters, model_dir='Output/Saved_Model/', use_mini_batch=False)
    else:
        print("Number of clusters must be a positive integer value!")

def trainKmeans():
    global num_iterations
    global kmeans
    global cluster_centers
    global pcaData
    global iterations_text
    kmeansTF()
    # train
    num_iterations= iterations_text.get("1.0",'end-1c')
    if num_iterations.isdigit():
        num_iterations = int(num_iterations)
        previous_centers=None
        for _ in range(num_iterations):
            kmeans.train(input_fn)
            cluster_centers = kmeans.cluster_centers()
            if previous_centers is not None:
                print ('delta: ', cluster_centers - previous_centers)
            previous_centers = cluster_centers
            print ('score: ', kmeans.score(input_fn))
        print ('cluster centers: ', cluster_centers)
        # list point to centroid 
        cluster_indices = list(kmeans.predict_cluster_index(input_fn))
        for i, point in enumerate(pcaData):
            cluster_index = cluster_indices[i]
            center = cluster_centers[cluster_index]
            print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)
        # Output Centroid locations
        with open("Output/KMeans_Centroids.txt", "w") as txt_file:
            txt_file.write(str(cluster_centers[0]))
            txt_file.write(str(cluster_centers[1]))
            txt_file.write(str(cluster_centers[2]))
        # Black points are centroids, colored points are all values closest to a centroid
        # C = value of which centroid a datapoint belongs to, each C has a unique color, represented on the graph via cmap
        fig = plt.figure(figsize = (10,10))
        plt.scatter(pcaData[:,0],pcaData[:,1], c=cluster_indices, cmap='rainbow')
        plt.scatter(cluster_centers[:,0] ,cluster_centers[:,1], color='black')
        fig.savefig('Output/KMeans_Clustered_Graph.png')
        plt.close(fig) 
    else:
        print("Number of iterations must be a positive integer value!")

def mapIndices(): 
    global pcaData
    global kmeans
    global cluster_centers
    # list point to centroid 
    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    # Colored points are all values closest to a centroid
    # C = value of which centroid a datapoint belongs to, each C has a unique color, represented on the graph via cmap
    fig = plt.figure(figsize = (10,10))
    plt.scatter(pcaData[:,0],pcaData[:,1], c=cluster_indices, cmap='rainbow')
    fig.savefig('Output/KMeans_Predicted_Graph.png')
    plt.close(fig) 
    
# Save model so training does not have to be repeated
def serving_input_receiver_fn():
    inputs = {}
    inputs = tf.placeholder(shape=[None, 2], dtype = tf.float32)
    return tf.estimator.export.TensorServingInputReceiver(inputs,inputs)

def exportModel():
    global kmeans
    kmeans.export_saved_model('Output/Saved_Model/', serving_input_receiver_fn)

def skipTrain():
    kmeansTF
    
def retrieve_input():
    global df_location
    input = self.myText_Box.get("1.0", 'end-1c')
    
root = Tk()
df_data = 0
standardizedData = 0
pcaData = 0
num_clusters = 0
num_iterations = 0
kmeans = 0
cluster_centers = 0
df_location = 0

root.geometry("500x500")

# Create buttons, specify which method they run using event listener (command=)
read_input_button = Button(root, text="Fetch Input",command=readInput)
preprocessing_button = Button(root, text="Apply Preprocessing",command=PCA) 
pca_plot_button = Button(root, text="Output PCA Scatter Plot",command=plotPCA)
elbow_button = Button(root, text="Output Elbow Graph",command=elbowGraph)
train_button = Button(root, text="Train Kmeans",command=trainKmeans)
predict_button = Button(root, text="Predict Profile",command=mapIndices)
output_model_button = Button(root, text="Output Model",command=exportModel)
load_model_button = Button(root, text="Load model ",command=kmeansTF)

# Create textbox for inputting dataset file path
input_text = Text(root)
iterations_text = Text(root)
clusters_text = Text(root)
input_text.insert(1.0, "Dataset file location")
iterations_text.insert(1.0, "Number of Iterations")
clusters_text.insert(1.0, "Number of Clusters")

# Add buttons to window
read_input_button.place(x=0, y=50)
preprocessing_button.place(x=0, y=80)
pca_plot_button.place(x=0, y=110)
elbow_button.place(x=0, y=140)
train_button.place(x=0, y=215)
predict_button.place(x=0, y=265)
output_model_button.place(x=0, y=315)
load_model_button.place(x=0, y=365)

# Add text and scales to window 
input_text.place(x=0, y=25, height=20, width=350)
clusters_text.place(x=0, y=165, height=20, width=350)
iterations_text.place(x=0, y=190, height=20, width=350)
# Run the mainloop of the program
root.mainloop()



