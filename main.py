"""
Author: Gananath R
DeepSpecta: Spatial and temporal classification of sequence data
Contact: https://github.com/gananath

#Powder Xrd data downloaded from http://rruff.info/
"""

from deepspectra import *
#from PIL import Image
import os
from sklearn.preprocessing import normalize

np.random.seed(2017)

#file path 
fpath='/media/user/data/crystal_system/'
epochs=250

#list of crystal systems    
nam=['Cubic',
     'Hexagonal',
     'Monoclinic',
     'Orthorhombic',
     'Rhombohedral',
     'Tetragonal',
     'Triclinic']


arr=[]
for fn in os.walk(fpath).next()[1]:
    for fn1 in os.listdir(fpath+fn+"/"):
        #extracting 2theta and intensity from csv files
        data=pd.read_csv(fpath+fn+"/"+fn1)
        data=data.ix[:,1:3]
        
        #normalizing columns
        data=normalize(data.values,axis=0)
        
        #convert sequence to 95x95 matrix with one zero padding in the edges
        #the end matrix has a shape of 97x97        
        newd=seq2tensor(data,95,95)
        
        #creting target one-hot encoded values
        Yval=np.zeros((1,len(nam)))
        idx=nam.index(fn)
        np.put(Yval,idx,1)
        
        #appending features and target to an array        
        arr.append([newd,Yval])
        print fn, fn1



arr=np.array(arr)
#randomizing [feature,target] 
np.random.shuffle(arr)


#extracting feature and targets from appended array
X=np.vstack(arr[:,0])
X=X.reshape(arr.shape[0],2,97,97)
y=np.vstack(arr[:,1])

#creating CNN model 
model=cnn_model(epochs)
print(model.summary())
model.fit(X,y,validation_split=0.33,batch_size=20,epochs=epochs,shuffle=True)
