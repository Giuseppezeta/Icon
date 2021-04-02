# Progetto Ingegneria della conoscenza

Studente: Giuseppe Cesare Zizzo

Matricola: 678603

# People Re-Identification by face

## Obiettivi del progetto

Questo progetto vuole unire gli insegnamenti del corso di Ingegneria della conoscenza con quelli ottenuti durante il percorso accademico. In particolare, ho posto la mia attenzione sull’utilizzo di tecniche di apprendimento non supervisionato unite con quelle di apprendimento supervisionato. Infatti, il sistema inizialmente non possiederà nessuna conoscenza e farà in modo che una tecnica di apprendimento supervisionato come la classificazione possa apprendere grazie agli esempi ricavati da una tecnica di apprendimento non supervisionato ovvero il clustering.  


## Introduzione

Il sistema sviluppato è basato sulla re-identificazione ovvero un task di face recognition, il task consiste nel riconoscere la presenza di uno stesso individuo in diversi filmati ripresi da un sistema di videosorveglianza. Andremo ad utilizzare alcune delle tecniche studiate durante il corso per creare un sistema che possa affrontare questo task. 
Il sistema si compone di due fasi:
-	Identificazione
-	Re-identificazione
Nella fase di Identificazione utilizzeremo il clustering, algoritmo non supervisionato, per raggruppare tutti i volti presenti nei vari filmati che costituiscono una parte del dataset in cui più individui compaiono più volte. I volti verranno quindi raggruppati e sulla base dei gruppi trovati ad ognuno di essi verrà associato un ID. Un gruppo quindi sarà proprio l'insieme delle facce di una stessa persona prese in più telecamere ed in momenti diversi. Nella fase di Re-identificazione invece andremo ad addestrare un classificatore con i volti ed i label trovati nella fase di Identificazione. In seguito, il classificatore sarà testato con dei nuovi volti estratti da nuovi filmati provenienti da una seconda parte del dataset.

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
- SVM (Support Vector Machine)

## Extra:
Oltre il clustering e la classificazione vengono implementate anche delle reti neurali convoluzionali (CNN) ma essendo un argomento fuori programma ne parlerò in questo paragrafo per tenere separate le cose senza però rischiare di non far comprendere l’elaborato.
Il sistema è in grado di acquisire volti e trasformali in vettori di caratteristiche proprio attraverso queste CNN. Le fasi in questione sono la Face detection e la Features Extraction. Nella face detection andiamo ad utilizzare una MTCNN (Multy Task CNN) che è pre addestrata per rilevare i volti presenti in un fotogramma. Questa restituisce i bounding box del volto che viene ritagliato per poi essere passato alla fase di Features Extraction. Quest’ultima invece, sempre con l’ausilio delle CNN, trasforma il volto in un vettore di caratteristiche. Per questa fase utilizziamo VGG-Face che è una rete neurale addestrata con milioni di facce.


## Dataset:
- ChokePoint Dataset http://arma.sourceforge.net/chokepoint/
- Sono stati testati altri dataset come CAVIAR e SALSA ma non sono riuscito a concludere il task efficacemente con questi a causa della bassa risoluzione


## Guida all’uso 

Il sistema presenta un insieme di tecniche per l’acquisizione delle facce da video e la loro elaborazione che sono la face detection e la features extraction. Queste fasi trattando argomenti fuori dal programma possono essere saltate utilizzando direttamente i due file .picke che contengono i vettori di caratteristiche estratti dalle facce trovate con la face detection e trasformate in vettori nella features extraction. Se si sceglie di utilizzare questi due file basta lasciar commentate le opportune parti del codice. Una volta fatto ciò il sistema andrà ad effettuare automaticamente la Re-identificazione e quindi andrà prima ad effettuare il clustering dei volti presenti, sotto forma di vettori, nel primo file (descriptors.p) ed andrà a trovare i labels. In seguito, addestrerà il classificatore con i vettori di caratteristiche ed i loro labels trovati. L’ultimo passo che il sistema effettua è proprio la re-identificazione dove usa i volti del secondo file (descriptors1.p) per effettuare un confronto attraverso il classificatore precedentemente addestrato. I risultati ottenuti sono una serie di varie combinazioni tra i vari metodi di clustering e classificazione.
Se si sceglie di non utilizzare i due file e caricare direttamente il dataset per testare le altre fasi allora accorrerà aggiornare manualmente i vettori contenenti la groundtruth  (y_true e y_true_test che sono rispettivamente corrispondenti a descriptors e descriptors1) e caricare i pesi per addestrare la rete neurale VGG.



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

---

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

---

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

LINK PER SCARICARE I PESI DI VGG:
https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view

