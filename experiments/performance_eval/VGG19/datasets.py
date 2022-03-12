import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100
import numpy as np

def MNIST(batch_size):
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    dataset = tf.data.Dataset.from_tensor_slices(
                (tf.cast(x_train, tf.float32), tf.cast(y_train,tf.int64)))
    dataset = dataset.shuffle(1000).batch(batch_size)

    return dataset, x_train, y_train, x_test, y_test


def CIFAR10(batch_size):
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # data preprocessing
    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)

    dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_train, tf.float32),
   tf.cast(y_train,tf.int64)))
    dataset = dataset.shuffle(1000).batch(batch_size)

    return dataset, x_train, y_train, x_test, y_test

def CIFAR100(batch_size):
    num_classes = 100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #dataset = tf.data.Dataset.from_tensor_slices(
  #(tf.cast(x_train, tf.float32),
   #tf.cast(y_train,tf.int64)))
    #dataset = dataset.shuffle(1000).batch(batch_size)

    gen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
    gen.fit(x_train)

    dataset = tf.data.Dataset.from_generator(lambda: gen.flow(x_train, y_train, batch_size=batch_size),
                 output_shapes=([None, 32,32,3], [None,100]), output_types=(tf.float32, tf.float32))

    return dataset, x_train, y_train, x_test, y_test