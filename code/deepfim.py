
# coding: utf-8

import pandas as pd
import src
from src.model.deepfim import DeepFIM
from tensorflow.python.keras.optimizers import Adam,Adagrad
from tensorflow.python.keras.optimizers import Adam,Adagrad
from tensorflow.python.keras.callbacks import EarlyStopping

train = pd.read_pickle('../avazu/enc_train.pkl')
val = pd.read_pickle('../avazu/enc_val.pkl')
feature_count = pd.read_pickle('../avazu/feature_dic.pkl')
target = ['click']
#feature_count.pop('userID')

sparse_feature_list = [src.SingleFeat(name,dim) for name,dim in feature_count.items()]

model = DeepFIM({'sparse':sparse_feature_list,'dense':[]})
model.compile('adam','binary_crossentropy',metrics=['binary_crossentropy'],)

count = -1
hist = model.fit([train[feat.name] for feat in sparse_feature_list],train[target].values,batch_size=4096,epochs=10,initial_epoch=0,validation_data=([val[feat.name] for feat in sparse_feature_list],val[target].values),verbose=1)

