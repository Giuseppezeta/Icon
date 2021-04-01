import cv2
import os
import time
from numpy import expand_dims
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from sklearn.cluster import DBSCAN

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import metrics

import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights(r'C:\Users\Giuseppe\Desktop\IC\vgg_face_weights.h5')

def Classification_Train(X_train,y_train,model = "KNN"):
  if model == "KNN":
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(X_train, y_train)
    """y_pred_knn_train = knn.predict(X_train)
    acc_knn_train = accuracy_score(y_train, y_pred_knn_train)
    print(f'KNN accuracy train = {acc_knn_train}')
    #X_train, X_test, y_train, y_test = train_test_split(np.array(X),np.array(y), test_size=0.20, random_state=42)"""
    return knn
  elif model == "SVM":
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        return clf
  elif model == "NB":
      gnb = GaussianNB()
      gnb.fit(X_train, y_train)
      return gnb
  elif model == "LR":
      clf = LogisticRegression(random_state=0).fit(X_train, y_train)
      return  clf
  elif model == "DT":
      clf = tree.DecisionTreeClassifier()
      clf = clf.fit(X_train, y_train)
      return clf

def Classification_Test(descriptors,y,MODEL):
    y_pred = MODEL.predict(descriptors)
    acc_test = accuracy_score(y, y_pred)
    f1 = f1_score(y,y_pred,average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    print(f'accuracy test = {acc_test}, precision test = {precision}, recall test = {recall}, F1 test = {f1}')

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def extract_face_from_image(image):
    #detector = MTCNN(steps_threshold=[0.8,0.9,0.9])
    detector = MTCNN(min_face_size=80, steps_threshold=[0.98,0.98,0.98])
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        conf = face['confidence']
        #cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), (255, 255, 0), 2)
        #x2, y2 = x1 + width, y1 + height
        face_images.append((x1, y1, width, height, conf))

    return face_images

def preprocess_image(img):
    #img = load_img(image_path, target_size=(224, 224))
    img = cv2.resize(img,(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_image_path(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def face_ex(path, out = r'\Users\Giuseppe\Desktop\outputfolder'):
  output_folder_path = os.path.join(out,'faces')
  os.makedirs(out)
  os.makedirs(output_folder_path)
  descriptors,images = face_det(path)
  n = 0
  img_paths = []
  for image in images:
    file_path = os.path.join(output_folder_path,"face_"+ str(n))
    new_path = file_path + '.jpg'
    cv2.imwrite(new_path, images[n])
    img_paths.append(new_path)
    n = n+1
  return descriptors, img_paths

def new_clust(descriptors , y_true, metodo = "dbscan", img_paths = None):
  new_imm_paths = [] # nuovi path
  d = [] #nuovi descrittori
  y = [] #predizioni
  descriptors = np.array(descriptors)
  #descriptors = StandardScaler().fit_transform(descriptors)
  if metodo == "dbscan":
    #db = DBSCAN(eps = 52, min_samples=2).fit(descriptors) #3facenet 32vgg con scaler  40vgg senza 52
    db = DBSCAN(eps = 0.163, min_samples=2, metric='cosine').fit(descriptors)
    labels = db.labels_
  elif metodo == "k-means":
    kmeans = KMeans(n_clusters=24, random_state=0).fit(descriptors)
    labels = kmeans.labels_
    #SE = kmeans.inertia_
    #print("K-means MSE con n_clusters =", k,SE/len(descriptors))
  elif metodo == "h":
      clustering = AgglomerativeClustering(n_clusters=24).fit(descriptors)
      labels = clustering.labels_
  num_classes = len(set(labels)) # Total number of clusters
  y_preds =  labels[labels != -1]
  core = descriptors[labels != -1]
  print("Number of clusters: {}".format(num_classes))
  for i in range(0, num_classes):
      indices = []
      class_length = len([label for label in labels if label == i])
      for j, label in enumerate(labels):
          if label == i:
              indices.append(j)
      print("Indices of images in the cluster {0} : {1}".format(str(i),str(indices)))
      print("Size of cluster {0} : {1}".format(str(i),str(class_length)))
      for k, index in enumerate(indices):
          if img_paths is not None:
            img = cv2.imread(img_paths[index])
            cv2.imshow(" " ,img)
            cv2.waitKey(0)
            new_imm_paths.append(img_paths[index])
          d.append(descriptors[index])
          #y.append(i)
          y.append(y_true[index])
  print("Homogeneity",metrics.homogeneity_score(labels, y_true))
  print("Completeness",metrics.completeness_score(labels, y_true))
  print("V_measure",metrics.v_measure_score(labels, y_true))
  return asarray(d), asarray(y), asarray(new_imm_paths)

def face_det(path):
  descriptors = []
  images = []
  for image in os.listdir(path):
    img = cv2.imread(os.path.join(path,image))
    print("estrazione facce")
    faces = extract_face_from_image(img)
    for (x,y,w,h,c) in faces:
      x, y = abs(x), abs(y)
      roi = img[y:y+h, x:x+w]
      images.append(roi)
      descriptors.append(vgg_face_descriptor.predict(preprocess_image(roi))[0])
      print("face detected")
      """cv2.imshow(" ",roi)"""
      print(c)
  return descriptors, images

def visualizzation(embedded, targets):
  X_embedded = TSNE(n_components=2).fit_transform(embedded)

  plt.figure(figsize=(10,10))

  for i, t in enumerate(set(targets)):
      idx = targets == t
      plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

  plt.legend(bbox_to_anchor=(1, 1))
  plt.show()

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findCosineDistance(source_representation, test_representation):
  a = np.matmul(np.transpose(source_representation), test_representation)
  b = np.sum(np.multiply(source_representation, source_representation))
  c = np.sum(np.multiply(test_representation, test_representation))
  return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def valutation(X_db, y_db, X_new, y_real, dist="c"):
    FAR = []
    DIR = []
    f1 = []
    acc = []
    if dist == "e":
        thresholds = np.arange(0, 130)
    elif dist == "c":
        thresholds = np.arange(0.0, 1.0, 0.01)
    thresholds = np.array(thresholds)
    for t in thresholds:
        y_pred = []
        tot = len(X_new)
        fp = 0
        tp = 0
        for x1 in range(len(X_new)):
            min_dist = 10000
            if y_real[x1] == -1:
                tot = tot - 1
            for x2 in range(len(X_db)):
                if dist == "e":
                    distance = findEuclideanDistance(X_new[x1], X_db[x2])
                elif dist == "c":
                    distance = findCosineDistance(X_new[x1], X_db[x2])
                if (distance < min_dist):
                    pos = x2
                    min_dist = distance
            if (min_dist < t):
                y_pred.append(y_db[pos])
                if y_db[pos] == y_real[x1]:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                y_pred.append(-1)
        f1.append(f1_score(y_real,y_pred,average='macro'))
        acc.append(accuracy_score(y_real,y_pred))

    opt_idx = np.argmax(f1)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]
    # Accuracy at maximal F1 score
    opt_acc = acc[opt_idx]

    plt.plot(thresholds, f1, label='f1 score');
    plt.plot(thresholds, acc, label='accuracy score');
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
    plt.xlabel('Distance threshold')
    plt.legend();
    plt.show()

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

output_folder = 'outputfolder'

train = r'\Users\Giuseppe\Desktop\IC\train'
test = r'\Users\Giuseppe\Desktop\IC\test'
out2 = r'\Users\Giuseppe\Desktop\outputfolder2'

y_true = [1,2,1,3,3,1,1,4,4,5,6,7,8,8,9,10,11,11,12,12,3,13,3,14,15,16,16,17,18,18,19,19,20,20,7,21,2,7,22,22,11,11,15,21,12,12,23,20,8,24,4,4,22,14,6,24,7,18,13,16,19,19,17,17,5,5,7,15,21,4,16,5,16,16,16,17,19,19,19,13,13,20,19,24,10,18,8,18,22,14,11,17,6,20,14,13,13,21,21,21,8,8,10,22,15,12]
y_true_test = [1,14,11,11,6,4,4,5,5,19,6,19,3,3,1,1,6,7,7,22,22,15,14,20,20,16,16,17,17,3,18,18,13,13,3,8,8,21,9,10,24,12,12,11,11,15,1,16,12,16,12,5,14,18,4,24,18,22,22,4,20,13,13,21,8,8,5,18,19,19,17,19,15,19,11,12,12,18,17,5,5,17,7,7,7,24,20,17,16,19,19,10,8,10,23,13,8,8,4,22,22,21,2,24,8,7,7,15,13,13,21,21,12,20]
y_true = asarray(y_true)
y_true_test = asarray(y_true_test)

""" Tenere commentata questa parte se non si vuole fare la face extraction che è un processo molto lento
start = time.time()
descriptors, img_paths = face_ex(train)
print("--- %s seconds ---" % (time.time() - start))
pickle.dump( descriptors, open( "descriptors.p", "wb" ) )
pickle.dump( img_paths, open( "img_paths.p", "wb" ) )
"""
descriptors = pickle.load( open( "descriptors.p", "rb" ) )
#img_paths = pickle.load( open( "img_paths.p", "rb" ) )

start = time.time()
desc, labels, paths = new_clust(descriptors, y_true)
desc1, labels1, paths1 = new_clust(descriptors, y_true,"k-means")
desc2, labels2, paths2 = new_clust(descriptors, y_true, "h")
print("--- %s seconds ---" % (time.time() - start))

""" Tenere commentata questa parte se non si vuole fare la face extraction del secondo set e se non si volgliono salvare i risultati del clustering

pickle.dump( desc, open( "desc.p", "wb" ) )
pickle.dump( desc1, open( "desc1.p", "wb" ) )
pickle.dump( desc2, open( "desc2.p", "wb" ) )
pickle.dump( labels, open( "labels.p", "wb" ) )
pickle.dump( labels1, open( "labels1.p", "wb" ) )
pickle.dump( labels2, open( "labels2.p", "wb" ) )
pickle.dump( paths, open( "paths.p", "wb" ) )
pickle.dump( paths1, open( "paths1.p", "wb" ) )

descriptors1, img_paths1 = face_ex(test,out2)
pickle.dump( descriptors1, open( "descriptors1.p", "wb" ) )
pickle.dump( img_paths1, open( "img_paths1.p", "wb" ) )

desc = pickle.load( open( "desc.p", "rb" ) )
labels = pickle.load( open( "labels.p", "rb" ) )
desc1 = pickle.load( open( "desc1.p", "rb" ) )
labels1 = pickle.load( open( "labels1.p", "rb" ) )
desc2 = pickle.load( open( "desc2.p", "rb" ) )
labels2 = pickle.load( open( "labels2.p", "rb" ) )
"""
descriptors1 = pickle.load( open( "descriptors1.p", "rb" ) )

visualizzation(desc, labels)
visualizzation(desc1, labels1)
visualizzation(desc2, labels2)

MODEL_DBSCAN_1 = Classification_Train(desc,labels,"KNN")
MODEL_DBSCAN_2 = Classification_Train(desc,labels,"SVM")
MODEL_DBSCAN_3 = Classification_Train(desc,labels,"LR")
MODEL_DBSCAN_4 = Classification_Train(desc,labels,"DT")
MODEL_DBSCAN_5 = Classification_Train(desc,labels,"NB")

MODEL_KMEANS_1 = Classification_Train(desc1,labels1,"KNN")
MODEL_KMEANS_2 = Classification_Train(desc1,labels1,"SVM")
MODEL_KMEANS_3 = Classification_Train(desc1,labels1,"LR")
MODEL_KMEANS_4 = Classification_Train(desc1,labels1,"DT")
MODEL_KMEANS_5 = Classification_Train(desc1,labels1,"NB")

MODEL_AGG_1 = Classification_Train(desc2,labels2,"KNN")
MODEL_AGG_2 = Classification_Train(desc2,labels2,"SVM")
MODEL_AGG_3 = Classification_Train(desc2,labels2,"LR")
MODEL_AGG_4 = Classification_Train(desc2,labels2,"DT")
MODEL_AGG_5 = Classification_Train(desc2,labels2,"NB")

print("DBSCAN TESTS")
print("KNN TEST")
Classification_Test(descriptors1,y_true_test,MODEL_DBSCAN_1)
print("SVM TEST")
Classification_Test(descriptors1,y_true_test,MODEL_DBSCAN_2)
print("LR TEST")
Classification_Test(descriptors1,y_true_test,MODEL_DBSCAN_3)
print("DT TEST")
Classification_Test(descriptors1,y_true_test,MODEL_DBSCAN_4)
print("NB TEST")
Classification_Test(descriptors1,y_true_test,MODEL_DBSCAN_5)

print("KMEANS TESTS")
print("KNN TEST")
Classification_Test(descriptors1,y_true_test,MODEL_KMEANS_1)
print("SVM TEST")
Classification_Test(descriptors1,y_true_test,MODEL_KMEANS_2)
print("LR TEST")
Classification_Test(descriptors1,y_true_test,MODEL_KMEANS_3)
print("DT TEST")
Classification_Test(descriptors1,y_true_test,MODEL_KMEANS_4)
print("NB TEST")
Classification_Test(descriptors1,y_true_test,MODEL_KMEANS_5)

print("Agglomerative Clustering TESTS")
print("KNN TEST")
Classification_Test(descriptors1,y_true_test,MODEL_AGG_1)
print("SVM TEST")
Classification_Test(descriptors1,y_true_test,MODEL_AGG_2)
print("LR TEST")
Classification_Test(descriptors1,y_true_test,MODEL_AGG_3)
print("DT TEST")
Classification_Test(descriptors1,y_true_test,MODEL_AGG_4)
print("NB TEST")
Classification_Test(descriptors1,y_true_test,MODEL_AGG_5)

