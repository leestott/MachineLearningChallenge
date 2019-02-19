#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# Being quite a lazy person, my natural inclination is to just bottleneck one of the standard CNNs and then train a small fully connected NN model... so we'll do that first.... probably deals with overfitting a bit better too...
# 
# Then we'll try fine tuning the same model.
# 
# Finally we'll build a basic CNN from scratch...
# 
# And finally finally we'll try one of Hinton's shiny new Capsule networks just for shits and giggles.

#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imread, imshow, show, subplot, title, get_cmap, hist
import numpy as np
from PIL import Image, ImageOps, ImageChops
import cv2
import os
import pickle
import itertools

#Inline Matplot graphics into the notebook
get_ipython().magic(u'matplotlib inline')


# From: https://gist.github.com/fabeat/6621507
# Using the version in the comments to the Gist
# Best practice default is using Bicubic rather than Antialias per http://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html#default-filter-for-thumbnails
def scale(image, max_size=(128,128), method=Image.BICUBIC):
    """
    resize 'image' to 'max_size' keeping the aspect ratio
    and place it in center of white 'max_size' image
    """
    image.thumbnail(max_size, method)
    offset = (int((max_size[0] - image.size[0]) // 2), int((max_size[1] - image.size[1]) // 2))
    back = Image.new("RGB", max_size, "white")
    back.paste(image, offset)

    return back






#%%
import shutil

shutil.rmtree('../gear_images_augmented')

import Augmentor
p = Augmentor.Pipeline("../gear_images",output_directory="../gear_images_augmented")
p.random_distortion(probability=.3, grid_width=4, grid_height=4, magnitude=8)
p.rotate(probability=.8, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.1, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.crop_random(probability=0.1, percentage_area=0.3)
p.sample(10000)


#%%
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image as kimage
from keras.applications.resnet50 import preprocess_input, decode_predictions

resnet50_model = ResNet50(weights='imagenet', include_top=False)


#%%
gear_images_dir = "../gear_images"
data = list()
hddata = list()
labels = list()
images = list()
features = list()
for directory in os.listdir(gear_images_dir):
    current_dir = gear_images_dir + '/' + directory
    print('Loading images from: {}'.format(current_dir))
    for imgName in os.listdir(os.fsencode(current_dir)):
        print('Loading: {}'.format(os.fsdecode(imgName)))
        image = Image.open(current_dir + '/' + os.fsdecode(imgName)) #Open as greyscale
        image = ImageOps.equalize(scale(image))
        images.append(image)
        data.append(np.asarray(image.convert('L')).flatten())
        labels.append(directory)
        
        #Featurize via ResNet50
        img = kimage.load_img(current_dir + '/' + os.fsdecode(imgName), target_size=(224, 224))
        x = kimage.img_to_array(img)
        hddata.append(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #features.append(resnet50_model.predict(x))
        features.append(1)
        
    print('Done')   


#%%



#%%
data_array = np.asarray(data)
print(data_array.shape)

hddata_array = np.asarray(hddata)
print(hddata_array.shape)

label_array = np.asarray(labels)
print(label_array.shape)

feature_array = np.asarray(np.squeeze(features))
print(feature_array.shape)

num_classes= len(list(set(labels)))
print(num_classes)


#%%



#%%
fig = plt.figure(figsize=(8, 6))
# plot several images
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(images[i], cmap=plt.cm.bone)
    


#%%
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
le.fit(label_array)
encoded_label_array = le.transform(label_array)


#%%


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_array,
        encoded_label_array, random_state=0)

X_feat_train, X_feat_test, y_feat_train, y_feat_test = train_test_split(feature_array,
        encoded_label_array, random_state=0)

X_hd_train, X_hd_test, y_hd_train, y_hd_test = train_test_split(hddata_array,
        encoded_label_array, random_state=0)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(X_feat_train.shape, X_feat_test.shape)
print(y_feat_train.shape, y_feat_test.shape)
print(X_hd_train.shape, X_hd_test.shape)
print(y_hd_train.shape, y_hd_test.shape)

#%% [markdown]
# Train a simple fully connected model on the features we bottlenecked out. We can see how much of a problem that overfitting is. By playing with the hyper-parameters; e.g. try changing numberof units in the layers.

#%%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = None
model = Sequential()
model.add(Dense(64,input_dim=len(X_feat_train[1,:])))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_feat_train, y_feat_train,
          epochs=50,
          batch_size=30,
          validation_data=(X_feat_test, y_feat_test))

#%% [markdown]
# Based on the Keras [applications sample](https://keras.io/applications/) but using ResNet and not inception

#%%
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = resnet50_model #As we already loaded the model above.

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- of size num_classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.summary()

# train the model on the new data for a few epochs
model.fit(X_hd_train, y_hd_train,
          epochs=20,
          batch_size=30,
          validation_data=(X_hd_test, y_hd_test))

#%% [markdown]
# Let's dump the layers out so that we can work out what we might unlock. Really good to refer to the [ResNet-50 model architechture](https://github.com/KaimingHe/deep-residual-networks#models)

#%%
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

#%% [markdown]
# Let's unlock from the Residual 5a branches up. So everything above the 4f Relu activation which is everything > 142

#%%
for layer in model.layers[:141]:
   layer.trainable = False
for layer in model.layers[142:]:
   layer.trainable = True

#Recompile and train low and slow.
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Now fine tune for 20 epochs.... again... overfitting is the big issue here
model.fit(X_hd_train, y_hd_train,
          epochs=20,
          batch_size=30,
          validation_data=(X_hd_test, y_hd_test))

#%% [markdown]
# Train a new, simple CNN. Based on some cut and paste code but changed to use different kernel dimesions (because we have multi-channel data)

#%%
X_hd_train.shape


#%%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import keras

input_shape = (224,224,3)

simple_model = Sequential()

simple_model.add(Conv2D(64, (3, 3),  input_shape=input_shape))
simple_model.add(Activation('relu'))
simple_model.add(MaxPooling2D(pool_size=(2, 2)))

simple_model.add(Conv2D(64, (3, 3)))
simple_model.add(Activation('relu'))
simple_model.add(MaxPooling2D(pool_size=(2, 2)))

simple_model.add(Conv2D(64, (3, 3)))
simple_model.add(Activation('relu'))
simple_model.add(MaxPooling2D(pool_size=(2, 2)))

simple_model.add(Conv2D(64, (3, 3)))
simple_model.add(Activation('relu'))
simple_model.add(MaxPooling2D(pool_size=(2, 2)))

simple_model.add(Flatten())

simple_model.add(Dense(220, activation='relu'))

simple_model.add(Dropout(0.5))
simple_model.add(Dense(num_classes, activation='softmax'))
simple_model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])



simple_model.summary()


from keras import callbacks
tbCallback = callbacks.TensorBoard(log_dir='SimpleCNN_Graph', histogram_freq=2,  
          write_graph=True, write_images=True)
tbCallback.set_model(simple_model)

#Then we can view with: tensorboard  --logdir SimpleCNN_Graph/

simple_model.fit(X_hd_train, y_hd_train,
          epochs=20,
          batch_size=30,
          validation_data=(X_hd_test, y_hd_test))


#%%


#%% [markdown]
# Finally we'll play with a simple CapsNet architechture...

#%%



