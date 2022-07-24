from keras.datasets import boston_housing
import pandas as pd

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