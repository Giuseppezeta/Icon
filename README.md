# Progetto Ingegneria della conoscenza

Studente: Giuseppe Cesare Zizzo
Matricola: 678603

# People Re-Identification by face

## Introduzione

Il sistema sviluppato permette la re-identificazione ovvero un task di face recognition, il task consiste nel riconoscere la presenza di uno stesso individuo in diversi filmati ripresi da un sistema di videosorveglianza. Andremo ad utilizzare alcune delle tecniche studiate durante il corso per creare un sitema che possa affrontare questo task.
Il sistema di compone di due fasi:

- Identificazione
- Reidentificazione

Nella fase di Identificazione utilizzeremo il clustering, algoritmo non supervisionato, per raggruppare tutti i volti presenti nei vari filmati che costituiscono una parte del dataset in cui più individue compaiono più volte. I volti verrano quindi ragruppati e sulla base dei gruppi trovati ad ognuni di essi verrà associato un ID. Un gruppo quindi sarà proprio l'insieme delle facce di una stessa persona prese in più telecamere ed in momenti diversi.
Nella fase di Re_identificazione invece andremo ad usare un classificatore addestrato con i volti ed i label di essi che andranno a costituire il training set. In seguito il classificatore sarà testato con dei nuovi volti estratti da nuovi filmati provenienti da una seconda parte del dataset.

## Algoritmi implementati

### Clustering:
- K-Means
- DBSCAN
- Agglomerative Clustering

### Classificazione:

- Knn
- Logistic Regression
- NaiveBayes
- Alberi
- SVM(SUPPORT VECTOR MACHINE)

## Dataset:
- ChokePoint Dataset

## Risultati:
### Clustering con K-means:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/clust_kmeans.PNG?raw=true)
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/Kmeans.PNG?raw=true)
### Clustering con DBSCAN:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/clust_dbscan.PNG?raw=true)
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/dbscan.PNG?raw=true)
### Agglomerative Clustering:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/clust_h.PNG?raw=true)
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/agglomerative.PNG?raw=true)

### Classificazione:
#### DBSCAN CON KNN:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/knn_dbscan.PNG?raw=true)
#### DBSCAN CON SVM:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/svm_dbscan.PNG?raw=true)
#### DBSCAN CON LR:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/LR_dbscan.PNG?raw=true)
#### DBSCAN CON DECISION TREE:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/dt_dbscan.PNG?raw=true)
#### DBSCAN CON NB:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/nb_dbscan.PNG?raw=true)

#### K-MEANS CON KNN:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/knn_kmeans.PNG?raw=true)
#### K-MEANS CON SVM:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/svm_kmeans.PNG?raw=true)
#### K-MEANS CON LR:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/LR_kmeans.PNG?raw=true)
#### K-MEANS CON DECISION TREE:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/dt_kmeans.PNG?raw=true)
#### K-MEANS CON NB:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/nb_kmeans.PNG?raw=true)

#### CLUSTERING GERARCHICO AGGLOMERATIVO CON KNN:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/agg_knn.PNG?raw=true)
#### CLUSTERING GERARCHICO AGGLOMERATIVO CON SVM:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/agg_svm.PNG?raw=true)
#### CLUSTERING GERARCHICO AGGLOMERATIVO CON LR:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/agg_LR.PNG?raw=true)
#### CLUSTERING GERARCHICO AGGLOMERATIVO CON DECISION TREE:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/agg_dt.PNG?raw=true)
#### CLUSTERING GERARCHICO AGGLOMERATIVO CON NB:
![alt text](https://github.com/Giuseppezeta/Icon/blob/main/Risultati/agg_nb.PNG?raw=true)

