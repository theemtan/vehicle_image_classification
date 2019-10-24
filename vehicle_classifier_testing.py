import numpy as np
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import pickle
import cv2
import random

X= pickle.load(open("X.pickle","rb"))
y= pickle.load(open("y.pickle","rb"))

IMG_SIZE=256
dict_test='D:\\datasets\\vehicle_test'

TEST = []

for img in os.listdir(dict_test):
    img_array = cv2.imread((os.path.join(dict_test, img)))
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # cv2.imshow('pepega',new_array)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    TEST.append(new_array)
    print('done')

TEST = np.array(TEST).reshape(-1,IMG_SIZE,IMG_SIZE,3)

new_model=tf.keras.models.load_model('vehicle_classifier_c.model')


# new_model.summary()
predictions = new_model.predict([TEST])
# print(predictions)
cat=['bus','plane','car','motorcycle','bicycle','public_train']

print(cat[np.argmax(predictions[00])])
plt.imshow(TEST[0])
plt.show()

# for i in np.arange(10):
#     g=random.randint(1,3000)
#     print(cat[np.argmax(predictions[g])])
#
#     plt.imshow(X[g])
#     plt.show()
