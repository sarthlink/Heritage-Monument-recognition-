import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pickle
import cv2
import joblib
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory("C:/Users/agarw/OneDrive/Desktop/minor1/Monument-Recognition/Dataset/Test_2",
                                              target_size = (64, 64),

                                                batch_size = 12,
                                                shuffle=True,
                                                class_mode = 'categorical')


dic={}
for i,j in test_set.class_indices.items():
	dic[j]=i



filename = 'monument_classification.sav'
img1 = image.load_img("C:/Users/agarw/OneDrive/Desktop/minor1/Monument-Recognition/Dataset/Predict/Badrinath_Temple.JPG",target_size = (64, 64))
filename = 'monument_classification.sav'
from keras.models import load_model
new_model = tf.keras.models.load_model(filename)
x=image.img_to_array(img1)
x=np.expand_dims(x,axis=0)

images = np.vstack([x])
classs = new_model.predict(images)
classes=np.argmax(classs,axis=1)
print()
print()
print("Your monument matches with this monument::")
print (dic[classes[0]])




'''
{'Aga Khan Palace': 0, 'Badrinath Temple': 1, 'Bekal': 2, 'Bhudha Temple': 3, 'Brihadeshwara Temple': 4, 'Cathederal': 5, 'Champaner': 6, 
'Chandi Devi mandir hariwar': 7, 'Cheese': 8, 'Chhatrapati Shivaji terminus': 9, 'Chittorgarh Padmini Lake Palace': 10, 'Daman': 11, 'Diu Museum': 12, 
'Fatehpur Sikri Fort': 13, 'Hampi': 14, 'Hoshang Shah Tomb': 15, 'India Gate': 16, 'Isarlat Sargasooli': 17, 'ajanta caves': 18, 'ajmeri gate delhi': 19, 
'albert hall museum': 20, 'bara imambara': 21, 'barsi gate hansi old': 22, 'basilica of bom jesus': 23, 'bharat mata mandir haridwar': 24, 'bhoramdev mandir': 25, 
'bidar fort': 26, 'buland darwaza': 27, 'byzantine architecture': 28, 'chandigarh college of architecture': 29, 'chapora fort': 30, 'charminar': 31, 
'chhatisgarh ke saat ajube': 32, 'chhatrapati shivaji statue': 33, 'chittorgarh': 34, 'city palace': 35, 'dhamek stupa': 36, 'diu': 37, 'dome': 38, 
'dubdi monastery yuksom sikkim': 39, 'falaknuma palace': 40, 'fatehpur sikri': 41, 'ford Auguda': 42, 'fortification': 43, 'gol ghar': 44, 'golden temple': 45,
 'hawa mahal': 46, 'hidimbi devi temple': 47, 'hindu temple': 48}
'''