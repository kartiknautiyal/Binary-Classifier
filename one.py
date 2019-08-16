import cv2 
import numpy as np 

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('C:/Users/Anil Nautiyal/Downloads/v_data/train', target_size=(64, 64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('C:/Users/Anil Nautiyal/Downloads/v_data/test',target_size=(64, 64),batch_size=32,class_mode='binary')

model.fit_generator(training_set, steps_per_epoch=25, epochs=20,validation_data=test_set, validation_steps=150)

