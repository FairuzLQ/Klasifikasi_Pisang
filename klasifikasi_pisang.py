import cv2 as cv
import glob
from skimage.feature import local_binary_pattern
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

radius = 2
n_points = 8 * radius
METHOD = 'uniform'

training = []
testing = []
responses = []
y_test = []

#ekstraksi 20 citra data training
for filename in glob.glob('uas_pcd/dataset_pisang2/train/pisang_ambon/*.jpeg'):
    #membaca gambar kelas 1 dan lakukan konversi ke grayscale, laku lakukan feature extraction dengan LBP
    data = local_binary_pattern(cv.imread(filename, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    #dapatkan histogram dari LBP untuk data kelas 1
    datahist,bins = np.histogram(data.ravel(),256,[0,256])
    #ubah vektor ke matriks dan lakukan transpose matriks untuk data kelas 1
    datahist = np.transpose(datahist[0:18,np.newaxis])
    training.extend(datahist)
    #hasil ekstraksi dimasukan kedalam dataframe
    dftraining = pd.DataFrame(training).astype(np.float32)
    responses.append("Ambon")

for filename in glob.glob('uas_pcd/dataset_pisang2/train/pisang_mas/*.jpeg'):
    #membaca gambar kelas 1 dan lakukan konversi ke grayscale, laku lakukan feature extraction dengan LBP
    data = local_binary_pattern(cv.imread(filename, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    #dapatkan histogram dari LBP untuk data kelas 2
    datahist,bins = np.histogram(data.ravel(),256,[0,256])
    #ubah vektor ke matriks dan lakukan transpose matriks untuk data kelas 2
    datahist = np.transpose(datahist[0:18,np.newaxis])
    training.extend(datahist)
    #hasil ekstraksi dimasukan kedalam dataframe
    dftraining = pd.DataFrame(training).astype(np.float32)
    responses.append("Mas")

for filename in glob.glob('uas_pcd/dataset_pisang2/train/pisang_tanduk/*.jpeg'):
    #membaca gambar kelas 1 dan lakukan konversi ke grayscale, laku lakukan feature extraction dengan LBP
    data = local_binary_pattern(cv.imread(filename, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    #dapatkan histogram dari LBP untuk data kelas 3
    datahist,bins = np.histogram(data.ravel(),256,[0,256])
    #ubah vektor ke matriks dan lakukan transpose matriks untuk data kelas 3
    datahist = np.transpose(datahist[0:18,np.newaxis])
    training.extend(datahist)
    #hasil ekstraksi dimasukan kedalam dataframe
    dftraining = pd.DataFrame(training).astype(np.float32)
    responses.append("Tanduk")

#ekstraksi citra data testing    
for filename in glob.glob('uas_pcd/dataset_pisang2/test/pisang_ambon/*.jpeg'):
    #membaca gambar kelas 1 dan lakukan konversi ke grayscale, laku lakukan feature extraction dengan LBP
    data = local_binary_pattern(cv.imread(filename, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    #dapatkan histogram dari LBP untuk data kelas 1
    datahist,bins = np.histogram(data.ravel(),256,[0,256])
    #ubah vektor ke matriks dan lakukan transpose matriks untuk data kelas 1
    datahist = np.transpose(datahist[0:18,np.newaxis])
    testing.extend(datahist)
    #hasil ekstraksi dimasukan kedalam dataframe
    dftesting = pd.DataFrame(testing).astype(np.float32)
    y_test.append("Ambon")
        
    
for filename in glob.glob('uas_pcd/dataset_pisang2/test/pisang_mas/*.jpeg'):
    #membaca gambar kelas 1 dan lakukan konversi ke grayscale, laku lakukan feature extraction dengan LBP
    data = local_binary_pattern(cv.imread(filename, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    #dapatkan histogram dari LBP untuk data kelas 2
    datahist,bins = np.histogram(data.ravel(),256,[0,256])
    #ubah vektor ke matriks dan lakukan transpose matriks untuk data kelas 2
    datahist = np.transpose(datahist[0:18,np.newaxis])
    testing.extend(datahist)
    #hasil ekstraksi dimasukan kedalam dataframe
    dftesting = pd.DataFrame(testing).astype(np.float32)
    y_test.append("Mas")
       
    
for filename in glob.glob('uas_pcd/dataset_pisang2/test/pisang_tanduk/*.jpeg'):
    #membaca gambar kelas 1 dan lakukan konversi ke grayscale, laku lakukan feature extraction dengan LBP
    data = local_binary_pattern(cv.imread(filename, cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
    #dapatkan histogram dari LBP untuk data kelas 3
    datahist,bins = np.histogram(data.ravel(),256,[0,256])
    #ubah vektor ke matriks dan lakukan transpose matriks untuk data kelas 3
    datahist = np.transpose(datahist[0:18,np.newaxis])
    testing.extend(datahist)
    #hasil ekstraksi dimasukan kedalam dataframe
    dftesting = pd.DataFrame(testing).astype(np.float32)   
    y_test.append("Tanduk")
       
    
#mencari nilai k-optimal
akurasi = 0
for i in range(2,16):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(dftraining,responses)
    score = cross_val_score(knn, dftraining, responses, cv=5, scoring='accuracy')
    score =score.mean()
    if(i%2==1):
        print("Cross Validation",i,":",score)
    if(score>akurasi and i%2==1):
        akurasi=score
        optimal = i
    
print()        
print("K-Optimal adalah",optimal)
print("Fold Validation",optimal,"sebesar",akurasi)
print()

#mengimplementasikan k-optimal pada knn untuk klasifikasi
knn=KNeighborsClassifier(n_neighbors = optimal) #define K=optimal
knn.fit(dftraining,responses)
res = knn.predict(dftesting)
print(res)
print(y_test)
print()

#menampilkan presentasi error prediksi
error=((y_test!=res).sum()/len(res))*100
print("Error prediksi = %.2f" %error, "%")

#menampilkan presentasi akurasi
akurasi = 100-error
print("Akurasi = %.2f" %akurasi,"%")

#mencetak confusion matrix
results = confusion_matrix(y_test, res) 
print ('Confusion Matrix :\n', results)
print ('Accuracy Score :',accuracy_score(y_test, res) )
print ('Report : ')
print (classification_report(y_test, res)) 

#menampilkan confusion matrix dengan plot
ax= plt.subplot()
sns.heatmap(results, annot=True, ax = ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Pisang Ambon', 'Pisang Mas','Pisang Tanduk']) 
ax.yaxis.set_ticklabels(['Pisang Ambon', 'Pisang Mas','Pisang Tanduk'])
plt.show()