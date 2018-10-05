
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, ImageDataGenerator
import os
import scipy


# ## **Train / Validation / Test**

# In[2]:


train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory('last_img_data/train/', target_size = (216,556), batch_size=9, class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory('last_img_data/validation/', target_size = (216,556), batch_size=4, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory('last_img_data/test/', target_size = (216,556), batch_size=4, class_mode='categorical')


# In[3]:


from keras import layers
from keras import models 

#model.add(layers.ZeroPadding2D(padding=(1, 1)))
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001)
#model.add(layers.Dropout(0.5))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='softmax'))
#model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


# In[51]:


# 음악을 시간 순으로 인식하니 1D convolution model으로 시도
def build_model_2():
    model = models.Sequential()
    ##
    #model.add(layers.Input((556,216,3)))
    #model.add(layers.ZeroPadding2D(padding=((2, 2), (0, 0))))
    model.add(layers.Conv2D(64, (72, 3),strides=(16, 1), activation='relu', input_shape=(216, 556, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    ##
    model.add(layers.Conv2D(128, (5, 3), activation='relu'))
    ##
    model.add(layers.ZeroPadding2D(padding=((0, 0), (1, 1))))#height/wide
    model.add(layers.Conv2D(128, (1, 5), activation='relu'))
    model.add(layers.MaxPooling2D((1, 3)))
    ##
    model.add(layers.ZeroPadding2D(padding=((0, 0), (1, 2))))
    model.add(layers.Conv2D(512, (1, 5), activation='relu'))
    model.add(layers.MaxPooling2D((1, 3)))
    ##
    #model.add(layers.Flatten())
    model.add(layers.Conv1D(3000, 3, strides=1, padding='valid',activation='relu')
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# In[52]:


model=build_model_2()


# In[53]:


model.summary()


# In[54]:


import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

history = model.fit_generator(train_generator, steps_per_epoch=1024, epochs=30, callbacks = [EarlyStopping(patience=5)], validation_data=validation_generator, validation_steps=512)


# In[55]:


model.save_weights('model_weights_2.h5')


# In[56]:


scores = model.evaluate_generator(test_generator, steps=400)


# In[57]:


scores


# In[78]:


import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12,4))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")

ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")

plt.suptitle("Loss and Accuracy")

