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

#Kod 3.27 K-fold Doğrulama
k=4
num_val_samples=len(train_data)//k
"""num_epochs=80
all_scores=[]
for i in range(k):
    print('processing fold #',i)
    #k ıncı parçadaki doprulama verisini hazırlar.
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    #Eğitim veri setini hazırlar: Veriler diğer parçalardan gelir.
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    model=build_model() #Keras modelini derler
    model.fit(partial_train_data,partial_train_targets,
            epochs=num_epochs,batch_size=1,verbose=0)
    #Doğrulama veri setinde değerlendirir.
    val_mse,val_mae=model.evaluate(val_data,val_targets,verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(f'{num_epochs} için oralama mae skoru: {np.mean(all_scores)}')"""

#Kod 3.28 Her parçada doğrulama günlüğünü kaydetmek
num_epochs=500
all_mae_histories=[]

for i in range(k):
    print('processing fold #',i)
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    model=build_model()
    history=model.fit(partial_train_data,partial_train_targets,
                        validation_data=(val_data,val_targets),
                        epochs=num_epochs,batch_size=1,verbose=0)
    mae_history=history.history['val_mae']
    all_mae_histories.append(mae_history)

#Kod 3.29 K-fold doğrulama skorlarının ortalamasını almak
average_mae_history=[np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#Kod 3.30 Doğrulama skorlarını çizdirmek. (Figure_1.png)
plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('Epoklar')
plt.ylabel("MAE-(Doğrulama)")
plt.show()

#Kod 3.31 Grafikte görülen farklı olan ilk 10 nokta veriden ayrılmıştır.
#Her noktayı önceki noktaların kayar ortalaması(moving average) olarak göstererek yumuşak bir eğri çizdirelim.

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()






