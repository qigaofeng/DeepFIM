
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import deepctr
from tensorflow.python.keras.optimizers import Adam,Adagrad


# In[2]:


train = pd.read_pickle('../avazu/enc_train.pkl')
val = pd.read_pickle('../avazu/enc_val.pkl')
feature_count = pd.read_pickle('../avazu/feature_dic.pkl')
target = ['click']
#feature_count.pop('userID')


# In[4]:


sparse_feature_list = [deepctr.SingleFeat(name,dim) for name,dim in feature_count.items()]


# In[5]:



from tensorflow.python.keras.optimizers import Adam,Adagrad
from tensorflow.python.keras.callbacks import EarlyStopping
model = deepctr.models.PNN({'sparse':sparse_feature_list,'dense':[]},embedding_size=16,hidden_size=[256,256,256],use_inner=False,use_outter=True,kernel_type='vec')
model.compile('adam','binary_crossentropy',metrics=['binary_crossentropy'],)

count = -1
hist = model.fit([train[feat.name] for feat in sparse_feature_list],train[target].values,batch_size=4096,epochs=10,initial_epoch=0,validation_data=([val[feat.name] for feat in sparse_feature_list],val[target].values),verbose=1)

