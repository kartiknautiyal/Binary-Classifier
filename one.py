import cv2 
import matplotlib.pyplot as plt

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
ep =20
test_set = test_datagen.flow_from_directory('C:/Users/Anil Nautiyal/Downloads/v_data/test',target_size=(64, 64),batch_size=32,class_mode='binary')

hist = model.fit_generator(training_set, steps_per_epoch=25, epochs=ep,validation_data=test_set, validation_steps=150)
val_acc = hist.history['val_acc']
acc= hist.history['acc']
x= list(range(1,ep+1))

plt.plot(x,acc,marker='o',label = 'acc')
plt.plot(x,val_acc,marker='o',label = 'val_acc')

plt.legend()
plt.show()
