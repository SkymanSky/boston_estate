import imp
from keras.datasets import boston_housing
import pandas as pd
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

#Kod 3.24 Boston dataseti yüklendi ve verilere bakıldı.
print(train_data.shape)
print(test_data.shape)
#print(train_targets)

df=pd.DataFrame(data=train_targets)
print(df.head())
print(df.describe())
print('--------------------------')
df_train_data=pd.DataFrame(data=train_data)
print(df_train_data.head())

#Kod 3.25 Verileri normalize(standardize) etmek.
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

#Yukarıdaki kodda test veri setini normalize etmek içinde eğitim veri setinde hesaplanan ortalama-
# ve standart sapma kullanılmıştır. İş akışında kesinlikle test veri seti üzerinde hesaplanmış bir
# değer kullanılmamalıdır.

#Kod 3.26 Model Tanımlama
#Çok az veri olduğundan 64 birimli 2 gizli katman kullanılacak.
#Genelde veri azaldıkça aşırı uydurma olasılığı artar bu sebeple küçük bir ağ kullanılır.
def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#Kod 3.32 Modelin son halini eğitmek.
model=build_model() #Derlenmiş yeni bir model kullanın.
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0) #Bu modeli tüm veri setinde eğitin.
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mae_score)






