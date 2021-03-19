# Icon
# Progetto Ingegneria della conoscenza

Studente: Giuseppe Cesare Zizzo
Matricola: 678603

# People Re-Identification by face

## Introduzione
-
Il sistema sviluppato permette la re-identificazione ovvero un task di face recognition, il task consiste nel riconoscere la presenza di uno stesso individuo in diversi filmati ripresi da un sistema di videosorveglianza. Andremo ad utilizzare alcune delle tecniche studiate durante il corso per creare un sitema che possa affrontare questo task.
Il sistema di compone di due fasi:

-Identificazione
-Reidentificazione

Nella fase di Identificazione utilizzeremo il clustering, algoritmo non supervisionato, per raggruppare tutti i volti presenti nei vari filmati che costituiscono una parte del dataset in cui più individue compaiono più volte. I volti verrano quindi ragruppati e sulla base dei gruppi trovati ad ognuni di essi verrà associato un ID. Un gruppo quindi sarà proprio l'insieme delle facce di una stessa persona prese in più telecamere ed in momenti diversi.
Nella fase di Re_identificazione invece andremo ad usare un classificatore addestrato con i volti ed i label di essi che andranno a costituire il training set. In seguito il classificatore sarà testato con dei nuovi volti estratti da nuovi filmati provenienti da una seconda parte del dataset.

## Algoritmi implementati

### Clustering:
K-Means
DBSCAN
Agglomerative Clustering

### Classificazione:

Knn
Logistic Regression
NaiveBayes
Alberi
SVM

Dataset:
ChokePoint Dataset

## Risultati:
-



Dataset ChokePoint
