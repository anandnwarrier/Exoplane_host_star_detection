import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

np.random.seed(1)

print('Loading train and test data...')
df1=pd.read_csv('exoTrain.csv')
#print(df1)
df2=pd.read_csv('exoTest.csv')
#print(df2)


"""
In the traning data, We have light intensites of stars measured at 3198 time instances. 
The training data has the flux sequenc for 5087 stars while the test data has the flux sequences for 570 stars. 
If the value in LABEL column is 2, it is an exoplanet host star and if it is 1, it is not an exoplanet host star.
"""

train_data=np.array(df1,dtype=np.float32)
#print(train_data)
test_data=np.array(df2,dtype=np.float32)
#print(test_data)

ytrain=train_data[:,0]
Xtrain=train_data[:,1:]

ytest=test_data[:,0]
Xtest=test_data[:,1:]

# print(ytrain,'\n',Xtrain)
# print(ytest,'\n',Xtest)

m=0   # A chosen exoplanet host star's index for plott
n=100 # A chosen non-exoplanet host star's index

#print('Shape of Xtrain:',np.shape(Xtrain),'\nShape of ytrain:',np.shape(ytrain))


plt.plot(Xtrain[m],'r')
plt.title('Light intensity vs time (for an exoplanet star)')
plt.xlabel('Time index')
plt.ylabel('Light intensity')
plt.show()

plt.plot(Xtrain[n],'b')
plt.title('Light intensity vs time (for a non exoplanet star)')
plt.xlabel('Time')
plt.ylabel('Light intensity')
plt.show()

###  Applying Fourier Transform

from scipy.fftpack import fft

print('Applying Fourier Transform...')

Xtrain=np.abs(fft(Xtrain,n=len(Xtrain[0]),axis=1))
Xtest=np.abs(fft(Xtest,n=len(Xtest[0]),axis=1))

# print(Xtrain,Xtrain.shape)

Xtrain=Xtrain[:,:1+int((len(Xtrain[0])-1)/2)]
# print('\n\n',Xtrain,Xtrain.shape)
#print('Shape of Xtrain:',np.shape(Xtrain),'\nShape of ytrain:',np.shape(ytrain))

Xtest=Xtest[:,:1+int((len(Xtest[0])-1)/2)]

plt.plot(Xtrain[m],'r')
plt.title('After FFT (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()

plt.plot(Xtrain[n],'b')
plt.title('After FFT (for a non exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()


#### Normalizing

from sklearn.preprocessing import normalize

print('Normalizing...')
Xtrain=normalize(Xtrain)
Xtest=normalize(Xtest)

plt.plot(Xtrain[m],'r')
plt.title('After FFT,Normalization (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()

plt.plot(Xtrain[n],'b')
plt.title('After FFT,Normalization (for a non exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()


#### Applying Gaussian Filter

from scipy import ndimage

print('Applying Gaussian filter...')
Xtrain=ndimage.filters.gaussian_filter(Xtrain,sigma=10)
Xtest=ndimage.filters.gaussian_filter(Xtest,sigma=10)

plt.plot(Xtrain[m],'r')
plt.title('After FFT,Normalization and Gaussian filtering (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()

plt.plot(Xtrain[n],'b')
plt.title('After FFT,Normalization and Gaussian filtering (for a non exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()


#### Scaling down the data

from sklearn.preprocessing import MinMaxScaler

print('Applying MinMaxScaler...')
scaler=MinMaxScaler(feature_range=(0,1))
Xtrain=scaler.fit_transform(Xtrain)
Xtest=scaler.fit_transform(Xtest)

plt.plot(Xtrain[m],'r')
plt.title('After FFT,Normalization, Gaussian filtering and scaling (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()

plt.plot(Xtrain[n],'b')
plt.title('After FFT,Normalization, Gaussian filtering and scaling (for a non exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()

print("By looking at the last 2 curves, we can say that fourier transform has helped us in seeing that, for an exoplanet star, the curve has a sudden dip.\
\n And, for the non-exoplanet star, the curve is almost on the same level with high fluctuations.")



#### LSTM RNN Model and Training

# reshaping to give as input to the RNN:
Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],1,Xtrain.shape[1]))
Xtest = np.reshape(Xtest,(Xtest.shape[0],1,Xtest.shape[1]))
#print('Shape of Xtrain:',np.shape(Xtrain),'\nShape of ytrain:',np.shape(ytrain))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# LSTM RNN Model:
def LSTM_RNN():
    model = Sequential()
    model.add(LSTM(32,input_shape=(1,Xtrain.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
    return model

estimator=KerasClassifier(build_fn=LSTM_RNN,epochs=10,batch_size=64,verbose=1)

# Training:
print('The model is being trained...')
history=estimator.fit(Xtrain,ytrain)


#### Training and Testing results

loss=history.history['loss']
acc=history.history['acc']

epochs=range(1,len(loss)+1)
plt.title('Training error with epochs')
plt.plot(epochs,loss,'bo',label='training loss')
plt.xlabel('epochs')
plt.ylabel('training error')
plt.show()

plt.plot(epochs,acc,'b',label='accuracy')
plt.title('Accuracy of prediction with epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# make predictions
trainPredict = estimator.predict(Xtrain,verbose=0)
testPredict = estimator.predict(Xtest,verbose=0)

plt.title('Training results')
plt.plot(trainPredict,'*',label='Predicted')
plt.plot(ytrain,'o',label='ground truth')
plt.xlabel('Train data sample index')
plt.ylabel('Predicted class (1 or 2)')
plt.legend()
plt.show()
print('We can see that the model is well trained on training data as it is predicting correctly')

plt.title('Performance of the model on testing data')
plt.plot(testPredict,'*',label='Predicted')
plt.plot(ytest,'o',label='ground truth')
plt.xlabel('Test data sample index')
plt.ylabel('Predicted class (1 or 2)')
plt.legend()
plt.show()

#### Accuracy, Precision and recall of the model

from sklearn import metrics as sk_met
accuracy_train=sk_met.accuracy_score(ytrain,trainPredict)
accuracy_test= sk_met.accuracy_score(ytest,testPredict)
print('\t\t train data \t test data')
print('accuracy:  ',accuracy_train,'\t',accuracy_test)

precision_train=sk_met.precision_score(ytrain,trainPredict)
precision_test=sk_met.precision_score(ytest,testPredict)
print('precision: ',precision_train,'\t',precision_test)

recall_train=sk_met.recall_score(ytrain,trainPredict)
recall_test=sk_met.recall_score(ytest,testPredict)
print('recall:    ',recall_train,'\t\t',recall_test)

