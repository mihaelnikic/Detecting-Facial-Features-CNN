from dataset import image_loader
from dataset.face_tracking_dataset import load_dataset
from fetches.i_optimizer import Optimizer
from fetches.loss.mse_loss import MeanSquaredErrorLoss
from fetches.metrics.regression.r_acc import RegressionAccuracy
from fetches.metrics.regression.r_squared import RSquared
from fetches.optimizers.rmsprop import RMSPropOptimizer
from net.layers.activation_layer import ActivationLayer
from net.layers.batch_norm_layer import BatchNormLayer
from net.layers.conv_layer import ConvolutionalLayer
from net.layers.dropout_layer import DropoutLayer
from net.layers.fc_layer import FullyConnectedLayer
from net.layers.flatten_layer import FlattenLayer
from net.layers.max_pool_layer import MaxPoolLayer
from net.network import Network
import tensorflow as tf
import utils.file_utils as futils
from placeholders.labels_placeholder import LabelsPlaceholder
from placeholders.lr.iterative_lr import IterativeLearningRate
from placeholders.picture_placeholder import PicturePlaceholder
import numpy as np
import random

FTRAIN = '/home/mihael/Documents/9. semestar/VIROKR/Projekt/Detecting-Facial-Features-CNN/dataset/kaggle/training.csv'

np.random.seed(100)
tf.set_random_seed(100)
random.seed(100)

# dataset specs
#X_train, y_train = load_dataset(fname=FTRAIN)
pic_width = 96
pic_height = 96
pic_channels = 1  # grayscale
num_classes = 15 * 2
# other
input = PicturePlaceholder(sample_input_shape=[pic_height, pic_width, pic_channels])
output = LabelsPlaceholder(num_classes=num_classes)

cnn = Network()
# First CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv1", filter_size=5, num_filters=24)) \
    .add_layer(MaxPoolLayer(name="pool1")) \
    .add_layer(BatchNormLayer(name="batch_norm1")) \
    .add_layer(ActivationLayer(name="relu1", activation_fn=tf.nn.relu))

# Second CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv2", filter_size=5, num_filters=36)) \
    .add_layer(MaxPoolLayer(name="pool2")) \
    .add_layer(BatchNormLayer(name="batch2")) \
    .add_layer(ActivationLayer(name="relu2", activation_fn=tf.nn.relu))

# Third CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv3", filter_size=5, num_filters=48)) \
    .add_layer(MaxPoolLayer(name="pool3")) \
    .add_layer(BatchNormLayer(name="batch3")) \
    .add_layer(ActivationLayer(name="relu3", activation_fn=tf.nn.relu))

# Fourth CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv4", filter_size=3, num_filters=64)) \
    .add_layer(MaxPoolLayer(name="pool4")) \
    .add_layer(BatchNormLayer(name="batch4")) \
    .add_layer(ActivationLayer(name="relu4", activation_fn=tf.nn.relu))

# Fifth CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv5", filter_size=2, num_filters=64,
                                 strides=[1, 1, 1, 1])) \
    .add_layer(BatchNormLayer(name="batch5")) \
    .add_layer(ActivationLayer(name="relu5", activation_fn=tf.nn.relu))

# Flatten layer
cnn.add_layer(FlattenLayer(name="flatten"))

# First FC layer
cnn.add_layer(FullyConnectedLayer(name="fc6", num_neurons=500)) \
    .add_layer(BatchNormLayer(name="batch6")) \
    .add_layer(ActivationLayer(name="relu6", activation_fn=tf.nn.relu)) \
    .add_layer(DropoutLayer(name="dout6", p_keep_train=0.8, p_keep_test=0.6))

# Second FC layer
cnn.add_layer(FullyConnectedLayer(name="fc7", num_neurons=90)) \
    .add_layer(BatchNormLayer(name="batch7")) \
    .add_layer(ActivationLayer(name="relu7", activation_fn=tf.nn.relu)) \
    .add_layer(DropoutLayer(name="dout7", p_keep_train=0.8, p_keep_test=0.6))

# Third FC layer
cnn.add_layer(FullyConnectedLayer(name="fc8", num_neurons=num_classes))

mse = MeanSquaredErrorLoss()
accuracy = RegressionAccuracy()
rsq = RSquared()
lr = IterativeLearningRate()
optimizer = RMSPropOptimizer(lr=lr)

cnn.build(input=input, output=output, optimizer=optimizer
          , loss=mse, metrics=[accuracy, rsq])

test_img, original_image, bbox = image_loader.read_image(image_file="/home/mihael/Documents/9. semestar/VIROKR/Projekt/Detecting-Facial-Features-CNN/dataset/slike/Ivo_Sanader.jpg",
                                   haarcascade_frontalface_file="/home/mihael/Documents/9. semestar/VIROKR/Projekt/Detecting-Facial-Features-CNN/dataset/haarcascade_frontalface_default.xml")

predicted = image_loader.plot_image(cnn, test_img, "/home/mihael/Documents/9. semestar/VIROKR/models/model.ckpt")
image_loader.plot_original_image(original_image, predicted, bbox)
