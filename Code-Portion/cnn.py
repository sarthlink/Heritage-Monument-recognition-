from keras.models import Sequential
import joblib
from keras.layers import Convolution2D
from keras.layers.core import Dropout
from keras.layers import MaxPooling2D
from keras import optimizers
from keras.layers import Flatten
import tensorflow as tf
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

model = Sequential()
model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1),padding="same",
                input_shape=(64,64,3),  activation='relu', data_format='channels_last'
                 ))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))


model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(units =49,activation='softmax'))

model.add(Dropout(0.4))



model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,

                                 horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("C:/Users/agarw/OneDrive/Desktop/minor1/Monument-Recognition/Dataset/Train_2",
                                                  target_size = (64, 64),
                                                  batch_size = 12,
                                                  shuffle=True,
                                                  class_mode = 'categorical')


test_set = test_datagen.flow_from_directory("C:/Users/agarw/OneDrive/Desktop/minor1/Monument-Recognition/Dataset/Test_2",
                                              target_size = (64, 64),

                                                batch_size = 12,
                                                shuffle=True,
                                                class_mode = 'categorical')

model.fit(                                  training_set,

                                            # steps_per_epoch = 10,

                                            epochs =50,

                                            validation_data = test_set)
# # save the model to disk
filename = 'monument_classification.sav'
model.save(filename)

print(test_set.class_indices)
































