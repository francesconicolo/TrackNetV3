
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.bouncer import bouncerDetector

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM

def f1Outlier(model_csv,video_name,file_path):
    
    #troviamo il csv con le tabelle di verità
    first_folder = Path(file_path).parts[0]
    truth_csv_path=str('trueLabels/'+str(first_folder)+'/'+str(video_name)+'_ball_true.csv')
    truth_csv=pd.read_csv(truth_csv_path)
    #inizializzazione variabili
    outlierTP=0
    outlierFN=0
    outlierFP=0
    # Ciclo sui frame nella tabella verità per valutare euristica di pulizia
    for _, truth_row in truth_csv.iterrows():
        if truth_row['Bounce'] == 0:  # Considera solo le righe con Bounce == 0 nel truth
            match_found = False
            for _, model_row in model_csv.iterrows():
                if model_row['Frame'] == truth_row['Frame'] and model_row['Outlier'] <= 0:
                    outlierTP += 1
                    match_found = True
                    break  # Interrompi il ciclo quando trovi la prima corrispondenza
            if not match_found:
                outlierFN += 1

    #Calcolo il numero di FP nell'outlier detector
    for _, model_row in model_csv.iterrows():
        if model_row['Outlier'] <= 0:  # Se il detector ha previsto un outlier
            #Ma il file verità dice che quel frame non è buono
            if model_row['Frame'] not in truth_csv['Frame'].values:
                outlierFP += 1
        
   # Calcolo Precision, Recall e F1 score con gestione della divisione per zero
    if (outlierTP + outlierFP) > 0:
        PrecisionOutlier = outlierTP / (outlierTP + outlierFP)
    else:
        PrecisionOutlier = 0  # O imposta su NaN o su un altro valore se preferisci

    if (outlierTP + outlierFN) > 0:
        RecallOutlier = outlierTP / (outlierTP + outlierFN)
    else:
        RecallOutlier = 0  # O imposta su NaN o su un altro valore se preferisci

    if (PrecisionOutlier + RecallOutlier) > 0:
        F1_score_outlier = 2 * ((PrecisionOutlier * RecallOutlier) / (PrecisionOutlier + RecallOutlier))
    else:
        F1_score_outlier = 0 
    # Limita a 6 cifre decimali
    PrecisionOutlier = round(PrecisionOutlier, 6)
    RecallOutlier = round(RecallOutlier, 6)
    F1_score_outlier = round(F1_score_outlier, 6)
    
    return PrecisionOutlier,RecallOutlier,F1_score_outlier

def f1Application(model_csv,video_name,file_path):
    #troviamo il csv con le tabelle di verità
    first_folder = Path(file_path).parts[0]
    truth_csv_path=str('trueLabels/'+str(first_folder)+'/'+str(video_name)+'_ball_true.csv')
    truth_csv=pd.read_csv(truth_csv_path)
    # Inizializza i contatori
    TP = 0  # Vero positivo
    FP = 0  # Falso positivo
    FN = 0  # Falso negativo
    # Inizializza i contatori PER ERRORI DI TRACKING

    Error_FP = 0  # Falso positivo
   
    # Ciclo sui frame nella tabella verità
    for _, truth_row in truth_csv.iterrows():
        if(truth_row['Bounce']==1):
            truth_frame = truth_row['Frame']
            frame_range = [truth_frame - 1, truth_frame, truth_frame + 1]

            # Cerca corrispondenze nel file del modello
            match_found = False
            for _, model_row in model_csv.iterrows():
                if model_row['Bounce'] > 0 and model_row['Frame'] in frame_range:
                    TP += 1
                    match_found = True
                    break

            if not match_found:
                FN += 1

    # Calcola i falsi positivi
    for _, model_row in model_csv.iterrows():
        if model_row['Bounce'] > 0:  # Se il modello ha previsto un bounce
            # Controlla se il frame del modello è fuori dal range della verità a terra
            # e che in truth_csv la colonna 'Bounce' sia 1
            if (model_row['Frame'] not in truth_csv[truth_csv['Bounce'] == 1]['Frame'].values and
                model_row['Frame'] - 1 not in truth_csv[truth_csv['Bounce'] == 1]['Frame'].values and
                model_row['Frame'] + 1 not in truth_csv[truth_csv['Bounce'] == 1]['Frame'].values):
                FP += 1

       #Calcolo il numero di errori in FP
    for _, model_row in model_csv.iterrows():
        if model_row['Bounce'] > 0:  # Se il modello ha previsto un bounce
            #Ma il file verità dice che quel frame non è buono
            if model_row['Frame'] in truth_csv['Frame'].values:
                truth_row = truth_csv[truth_csv['Frame'] == model_row['Frame']].iloc[0]
                if(truth_row['Bounce']==0):
                    Error_FP += 1

    if (TP + FP) > 0:
        Precision = TP / (TP + FP)
    else:
        Precision = 0

    if (TP + FN) > 0:
        Recall = TP / (TP + FN)
    else:
        Recall = 0

    if (Precision + Recall) > 0:
        F1_score = 2 * ((Precision * Recall) / (Precision + Recall))
    else:
        F1_score = 0

    #score senza considerare i nuovi errori
    if (TP + (FP - Error_FP)) > 0:
        new_precision = TP / (TP + (FP - Error_FP))
    else:
        new_precision = 0

    if (new_precision + Recall) > 0:
        new_F1_score = 2 * ((new_precision * Recall) / (new_precision + Recall))
    else:
        new_F1_score = 0

    # Limita a 6 cifre decimali
    Precision = round(Precision, 6)
    Recall = round(Recall, 6)
    F1_score = round(F1_score, 6)

    totalFrames=len(model_csv)
    wrongFrames=len(truth_csv[truth_csv['Bounce'] == 0])

    return Precision,Recall,F1_score,new_precision,new_F1_score,totalFrames,wrongFrames



def correctBounces(predicted_csv):
    windowSize=8
    threshold=4
    # Scorri il DataFrame per esaminare ogni intervallo di windowSize
    for i in range(1,len(predicted_csv) - windowSize):
        # Seleziona il sottogruppo di frame nel range corrente
        window = predicted_csv.iloc[i:i + windowSize]
        # Conta quanti bounces ci sono nel range
        bounce_count = (window['Bounce'].ne(0)).sum()
        # Se i bounces superano la soglia, correggi i bounces a 0
        if bounce_count >= threshold:
            predicted_csv.loc[i:i + windowSize, 'Bounce'] = 0
            predicted_csv.loc[i:i + windowSize, 'Outlier'] = -7
        if(predicted_csv.loc[i-1,'Outlier']==-7):
            predicted_csv.loc[i,'Bounce']=0
        i=i+windowSize
    return predicted_csv

def customOutlierDetector(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    
    # Estrai le coordinate X e Y
    predicted_X = predicted_csv['X'].values
    predicted_Y = predicted_csv['Y'].values
    lenPred = len(predicted_X)
    
    # Inizializza la lista degli outliers con zeri
    outliers = [0] * lenPred
    
    # Impostazione della soglia per il cambiamento significativo (può essere modificato se necessario)
    threshold_distance=130
    # Variabile per tracciare l'ultimo valore non outlier (frame coerente)
    last_good_outlier_idx = 13  # Iniziamo con il primo frame come "coerente"
    counter_outliers=0
    outliers[0:last_good_outlier_idx]=[1] * last_good_outlier_idx
    # Itera sui dati a partire dal secondo frame (i=5) fino all'ultimo frame
    for i in range(13, lenPred):
         # Se X e Y sono entrambi 0, considera il frame come non outlier (1)
        if predicted_X[i] == 0 and predicted_Y[i] == 0:
            outliers[i] = 1
            last_good_outlier_idx = i  # Ripristina il frame corrente come "coerente"
        else:
            distance =  math.sqrt((predicted_X[i]-predicted_X[last_good_outlier_idx])**2 + (predicted_Y[i]-predicted_Y[last_good_outlier_idx])**2)
            # Se c'è una differenza significativa in distanza AB, segna come outlier
            if distance>threshold_distance:
                #Se l'ultimo inlier è un fuori visione, e la Y attuale è superiore a 200,sarà tutto negativo
                if predicted_Y[last_good_outlier_idx]==0 and predicted_Y[i]>130:
                    outliers[i]=-7
                    counter_outliers=0
                #Se la pallina precedente è un fuori visione, e la Y attuale è inferiore a 200, segno il nuovo inlier
                elif predicted_Y[last_good_outlier_idx]==0 and predicted_Y[i]<130:
                    outliers[i]=1
                    last_good_outlier_idx = i  # Ripristina il frame corrente come "coerente"
                    counter_outliers=0 #resetto gli outliers
                else:
                    outliers[i] = -7  # Marca come outlier il frame corrente
                    counter_outliers=counter_outliers+1 #conto gli outliers
            else:
                outliers[i] = 1  # Frame normale, non outlier
                last_good_outlier_idx = i  # Ripristina il frame corrente come "coerente"
                counter_outliers=0 #resetto gli outliers
            #se individuo 8 outliers di fila, probabilmente l'algoritmo sta sbagliando e quindi segno buono il frame corrente
            if(counter_outliers>8):
                # for j in range(i-8,i):
                #     outliers[j]=1
                # for j in range(i-16,i-8):
                #     outliers[j]=-7
                counter_outliers=0
                last_good_outlier_idx=i
        
    # Aggiungi la colonna 'Outliers' al DataFrame
    predicted_csv['Outlier'] = outliers
    
    # Restituisce il DataFrame con i risultati
    return predicted_csv

def dbscan(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    # Seleziona le coordinate X, Y e Frame per l'analisi
    Coordinate = predicted_csv[['X', 'Y', 'Frame']]
    
    # Normalizzazione dei dati
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    
    # Configura e applica DBSCAN
    dbscan = DBSCAN(eps=0.45, min_samples=8)
    outliers = dbscan.fit_predict(Coordinate_scaled)
    
    # Aggiungi la colonna 'Outlier' 
    predicted_csv['Outlier'] = outliers

    return predicted_csv

def isolationForest(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X','Y','Frame']]
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    # Configura Isolation Forest
    iso_forest = IsolationForest(contamination=0.2,random_state=42)
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
    Coordinate = predicted_csv[['X', 'Y', 'Frame']]

    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    
    # Uso del OneClassSVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    outliers = ocsvm.fit_predict(Coordinate_scaled)  # -1 indica outlier, 1 indica inlier
    predicted_csv['Outlier'] = outliers
    return predicted_csv

def knn(predicted_csv):
    predicted_csv = pd.DataFrame(predicted_csv)
    Coordinate = predicted_csv[['X', 'Y', 'Frame']]
    
    # Normalizzazione
    scaler = StandardScaler()
    Coordinate_scaled = scaler.fit_transform(Coordinate)
    
    # KNN con k=10 vicini
    nbrs = NearestNeighbors(n_neighbors=8)
    nbrs.fit(Coordinate_scaled)
    
    # Calcolo delle distanze ai vicini
    distances = nbrs.kneighbors(Coordinate_scaled)
    
    # Distanza media
    avg_distances = distances.mean(axis=1)
    
    # Calcolare la soglia come 90° percentile
    threshold = np.percentile(avg_distances, 95)
    
    # Considerare outlier i punti con distanza maggiore della soglia
    predicted_csv['Outlier'] = np.where(avg_distances > threshold, -1, 1)
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

def dataAnalysis(predicted_csv,video_name,preset,save_dir,algo='custom'):
    print('Default custom outlier: ', end='')
    predicted_csv=customOutlierDetector(predicted_csv)
    predicted_csv['Bounce']=[0]*len(predicted_csv)
    rows_with_outlier_minus7 = predicted_csv[predicted_csv['Outlier'] == -7].reset_index()
    predicted_csv_filtered = predicted_csv[predicted_csv['Outlier'] != -7].reset_index()
    print('Done.')
    print(f'selected: {algo} outlier ',end='')
    data=pd.DataFrame(predicted_csv_filtered)
    if algo=='isolationForest':
        data=isolationForest(predicted_csv_filtered)
    elif algo=='dbscan':
        data=dbscan(predicted_csv_filtered)
    elif algo=='lof':
        data=lof(predicted_csv_filtered)
    elif algo=='kMeans':
        data=kMeans(predicted_csv_filtered)
    elif algo=='oneClassSVM':
        data=oneClassSVM(predicted_csv_filtered)
    elif algo=='knn':
        data=oneClassSVM(predicted_csv_filtered)
    print(f'Done.')
    print(f'Bouncer detector: ',end='')
    data=bouncerDetector(data,preset)
    # Visualizzazione dei dati
    print(f'Done')
    #ulteriore pulizia
    print(f'Pulizia bouncer: ',end='')
    data=correctBounces(data)
    print(f'Done')
    final_result = pd.concat([data, rows_with_outlier_minus7], axis=0).sort_values(by='Frame').reset_index(drop=True)

    #applicazione dello scorer
    Precision, Recall, F1_score, new_precision, new_F1_score, totalFrames, wrongFrames = f1Application(data, video_name, save_dir)
    
    scoreEuristica=Precision, Recall, F1_score, new_precision, new_F1_score
    percentScoreTracknet=(1-(wrongFrames/totalFrames))*100
    ScoreTrackNet=totalFrames,wrongFrames,percentScoreTracknet
    scoreOutlier=f1Outlier(final_result,video_name,save_dir)
    # printGraph(final_result,algo,video_name)
    
    return final_result,scoreEuristica,scoreOutlier,ScoreTrackNet