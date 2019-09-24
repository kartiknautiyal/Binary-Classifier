import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
valid_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('C:/Users/Anil Nautiyal/Downloads/v_data/train', target_size=(64, 64),batch_size=20,class_mode='binary', shuffle = True)
valid_generator = valid_datagen.flow_from_directory('C:/Users/Anil Nautiyal/Downloads/v_data/valid',target_size=(64, 64),batch_size=10,class_mode='binary',shuffle= True)
test_generator = test_datagen.flow_from_directory('C:/Users/Anil Nautiyal/Downloads/v_data/test', target_size=(64,64), batch_size =1, class_mode = 'binary', shuffle = False)


ep=50
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


hist = model.fit_generator(generator = train_generator, steps_per_epoch=12, epochs=ep,validation_data=valid_generator, validation_steps=3)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST, verbose=1)

predicted_class_indices=np.around(pred)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predicted_class_indices = np.ndarray.flatten(predicted_class_indices)
predictions = [labels[k] for k in predicted_class_indices]


filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

val_acc = hist.history['val_acc']
acc= hist.history['acc']
x= list(range(1,ep+1))

plt.plot(x,acc,marker='o',label = 'acc')
plt.plot(x,val_acc,marker='o',label = 'val_acc')

plt.legend()
plt.show()
