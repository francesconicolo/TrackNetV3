
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM

def dbscan(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X','Y','Frame']]
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    # Configura DBSCAN
    dbscan = DBSCAN()
    outliers = dbscan.fit_predict(Coordinate_scaled)
    predicted_csv['Outlier'] = outliers  
    return predicted_csv

def isolationForest(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X','Y','Frame']]
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    # Configura Isolation Forest
    iso_forest = IsolationForest(contamination=0.1,random_state=42)
    outliers = iso_forest.fit_predict(Coordinate_scaled)
    # Aggiungi il risultato al dataset
    predicted_csv['Outlier'] = outliers  # -1 = outlier, 1 = normale
    return predicted_csv

def lof(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X','Y','Frame']]
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    lof = LocalOutlierFactor(n_neighbors=20)
    # Calcola i punteggi LOF (-1 indica un outlier)
    outliers = lof.fit_predict(Coordinate_scaled)
    predicted_csv['Outlier'] = outliers
    return predicted_csv

def kMeans(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X','Y','Frame']]
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    # K-Means clustering
    kmeans = KMeans() 
    kmeans.fit(Coordinate_scaled)
    # Calcolo delle distanze dai centroidi
    distances = np.linalg.norm(Coordinate_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    # Determinazione della soglia per gli outlier (95° percentile delle distanze)
    threshold = np.percentile(distances, 95)
    predicted_csv['Outlier'] = np.where(distances > threshold, -1, 1) 
    return predicted_csv

def oneClassSVM(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X','Y','Frame']]
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    #uso del oneClassSVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    outliers = ocsvm.fit_predict(Coordinate_scaled)  # -1 indica outlier, 1 indica inlier
    predicted_csv['Outlier'] = outliers
    return predicted_csv

def knn(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X','Y','Frame']]
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    nbrs = NearestNeighbors(n_neighbors=5)
    nbrs.fit(Coordinate_scaled)
    # Calcolo delle distanze ai k vicini più prossimi
    distances = nbrs.kneighbors(Coordinate_scaled)
    # Calcolare la distanza media dai k vicini più prossimi per ogni punto
    avg_distances = distances.mean(axis=1)
    # Determinazione della soglia per gli outlier (puoi cambiare questa logica a seconda dei tuoi dati)
    threshold = np.percentile(avg_distances, 90)  # 90° percentile come soglia per outlier
    # Considera come outlier i punti con distanza media maggiore della soglia
    predicted_csv['Outlier'] = np.where(avg_distances > threshold, -1, 1)  # -1 indica outlier, 1 indica normale
    return predicted_csv

def printGraph(elaborated_csv,algo,video_name):
    # Separazione dei dati normali e outlier
    normal_data = elaborated_csv[elaborated_csv['Outlier'] != -1]
    outlier_data = elaborated_csv[elaborated_csv['Outlier'] == -1]
    plt.figure(figsize=(8, 6))
    plt.scatter(normal_data['X'], normal_data['Y'], label='Normali', color='green')
    plt.scatter(outlier_data['X'], outlier_data['Y'], label='Outlier', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Rilevamento degli Outlier con '+ str(algo))
    # Salvataggio dell'immagine
    plt.savefig('./prediction/'+video_name+'/'+video_name+'_'+str(algo)+'.png')  # Puoi cambiare il formato se desideri, ad esempio .jpg, .pdf, etc.
    plt.close()  # Chiude il grafico per liberare risorse

def dataAnalysis(predicted_csv,video_name,algo='isolationForest'):
    if algo=='isolationForest':
        data=isolationForest(predicted_csv)
    elif algo=='dbscan':
        data=dbscan(predicted_csv)
    elif algo=='lof':
        data=lof(predicted_csv)
    elif algo=='kMeans':
        data=kMeans(predicted_csv)
    elif algo=='oneClassSVM':
        data=oneClassSVM(predicted_csv)
    elif algo=='knn':
        data=oneClassSVM(predicted_csv)
    # Visualizzazione dei dati
    printGraph(data,algo,video_name)

    return data