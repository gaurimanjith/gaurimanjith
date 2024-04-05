import tensorflow as tf
import keras.api._v2.keras as keras
import os
print(tf.__file__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
Image_Size = 256
Batch_Size = 65
channels = 3
epochs = 50
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Plant_dataset",
    shuffle=True,
    image_size = (Image_Size,Image_Size),
    batch_size = Batch_Size,
)
classnames = dataset.class_names
#for loading the images into the plot and showing it on the output screen:-
for image_batch, lable_batch in dataset.take(1):
    plt.figure(figsize=(10,10))
    for i in range(12):
        #print(image_batch, lable_batch)
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.axis("off")
        plt.title(classnames[lable_batch[i]])
plt.show()


def get_dataset_partition(ds, train_split = 0.8, val_split = 0.1, test_split = 0.1, shuffle = True, shuffle_size = 4000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    ds_size = len(ds)
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partition(dataset)  
print(len(train_ds),len(val_ds),len(test_ds))

#buffer_siz = tf.data.AUTOTUNE
shuffle_size = 4000

'''
train_ds.cache().shuffle().prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds.cache().shuffle().prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds.cache().shuffle().prefetch(buffer_size = tf.data.AUTOTUNE)'''


train_ds = train_ds.cache().shuffle(buffer_size=shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(buffer_size=shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(buffer_size=shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)


resize_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(Image_Size, Image_Size),
    layers.experimental.preprocessing.Rescaling(1.0/255.0)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

input_shape = (Batch_Size, Image_Size, Image_Size, channels)
n_classes = 24

model = models.Sequential([
    resize_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)
print(model.summary())

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=epochs,
    batch_size=Batch_Size,
    verbose=1,
    #validation_batch_size=val_ds
    validation_data=val_ds
)

scores = model.evaluate(test_ds)
print(scores)
#history.params


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

def prediction(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img[i].numpy())
    img_array = tf.expand_dims(img_array,0)

    predictions = model.predict(img_array)
    predicted_class = classnames[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence
'''
for images, label_batch in test_ds.take(1):  # Adjust to get the desired number of batches
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Display the first 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = prediction(model, images[i].numpy())
        truth = 1
        if confidence < 75:
            truth = 0
        if truth == 1:
            plt.title(f"{predicted_class}-fake")
        else:
            plt.title(f"{predicted_class}-true")
        plt.axis("off")
    plt.show()
'''
model_version =2
model.save(f"../Model/{model_version}")