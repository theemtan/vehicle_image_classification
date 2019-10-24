import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import pickle

DIC = 'D:\\datasets\\vehicle_'
CAT = ['bus','plane','car','motorcycle','bicycle','public_train']
IMG_SIZE = 128

training_data=[]

def create_training_data():

    for cat in CAT:
        path = os.path.join(DIC,cat)
        class_num = CAT.index(cat)
        for img in os.listdir(path):
            img_array = cv2.imread((os.path.join(path,img)))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array,class_num])

create_training_data()
print(len(training_data))

import random
random.seed(1001)
random.shuffle(training_data)

X = []
y = []

for features ,label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

pickle_out = open('X_V1.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open('y_V1.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()

print(X.shape)
