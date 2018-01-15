from dataset import image_loader
from fetches.loss.mse_loss import MeanSquaredErrorLoss
from fetches.metrics.regression.r_acc import RegressionAccuracy
from fetches.metrics.regression.r_squared import RSquared
from fetches.optimizers.rmsprop import RMSPropOptimizer
from net.layers.activation_layer import ActivationLayer
from net.layers.batch_norm_layer import BatchNormLayer
from net.layers.conv_layer import ConvolutionalLayer
from net.layers.fc_layer import FullyConnectedLayer
from net.layers.global_avg_pool_layer import GlobalAveragePoolLayer
from net.layers.max_pool_layer import MaxPoolLayer
from net.network import Network
import tensorflow as tf
from placeholders.labels_placeholder import LabelsPlaceholder
from placeholders.lr.iterative_lr import IterativeLearningRate
from placeholders.picture_placeholder import PicturePlaceholder
import argparse
import sys

# dataset specs
pic_width = 96
pic_height = 96
pic_channels = 1  # grayscale
num_classes = 15 * 2
# other
input = PicturePlaceholder(sample_input_shape=[pic_height, pic_width, pic_channels])
output = LabelsPlaceholder(num_classes=num_classes)

cnn = Network()
# First CNN layer
cnn.add_layer(BatchNormLayer(name="batch_norm1")) \
    .add_layer(ConvolutionalLayer(name="conv1", filter_size=5, num_filters=24, strides=[1, 1, 1, 1])) \
    .add_layer(ActivationLayer(name="relu1", activation_fn=tf.nn.relu)) \
    .add_layer(MaxPoolLayer(name="pool1", padding="VALID"))

# Second CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv2", filter_size=5, num_filters=36, strides=[1, 1, 1, 1], padding="VALID")) \
    .add_layer(ActivationLayer(name="relu2", activation_fn=tf.nn.relu)) \
    .add_layer(MaxPoolLayer(name="pool2", padding="VALID"))

# Third CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv3", filter_size=5, num_filters=48, strides=[1, 1, 1, 1], padding="VALID")) \
    .add_layer(ActivationLayer(name="relu3", activation_fn=tf.nn.relu)) \
    .add_layer(MaxPoolLayer(name="pool3", padding="VALID"))

# Fourth CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv4", filter_size=3, num_filters=64, strides=[1, 1, 1, 1], padding="VALID")) \
    .add_layer(ActivationLayer(name="relu4", activation_fn=tf.nn.relu)) \
    .add_layer(MaxPoolLayer(name="pool4"))

# Fifth CNN layer
cnn.add_layer(ConvolutionalLayer(name="conv5", filter_size=2, num_filters=64, strides=[1, 1, 1, 1], padding="VALID")) \
    .add_layer(ActivationLayer(name="relu5", activation_fn=tf.nn.relu))

# Global avg pooling layer
cnn.add_layer(GlobalAveragePoolLayer(name="gapool1"))

# First FC layer
cnn.add_layer(FullyConnectedLayer(name="fc6", num_neurons=500)) \
    .add_layer(ActivationLayer(name="relu6", activation_fn=tf.nn.relu)) \
 \
# Second FC layer
cnn.add_layer(FullyConnectedLayer(name="fc7", num_neurons=90)) \
    .add_layer(ActivationLayer(name="relu7", activation_fn=tf.nn.relu)) \
 \
# Third FC layer
cnn.add_layer(FullyConnectedLayer(name="fc8", num_neurons=num_classes))

mse = MeanSquaredErrorLoss()
accuracy = RegressionAccuracy()
rsq = RSquared()
lr = IterativeLearningRate()
optimizer = RMSPropOptimizer(lr=lr)

cnn.build(input=input, output=output, optimizer=optimizer
          , loss=mse, metrics=[accuracy, rsq])

parser = argparse.ArgumentParser(description='Detekcija ključnih točaka na slici.')
parser.add_argument(
        '-s',
        '--Slika',
        help='Datoteka sa slikom.',
        type=str,
    )
parser.add_argument('-v', help='Vrati nazad sliku u izvorni oblik.', action='store_true')
args = parser.parse_args()
if args.Slika is None:
    print("Potrebno je specificirati datoteku sa slikom!", file=sys.stderr)
    sys.exit(-1)

test_img, original_image, bbox = image_loader.read_image(
    image_file=args.Slika,
    haarcascade_frontalface_file="../dataset/haarcascade_frontalface_default.xml")

predicted = image_loader.plot_image(cnn, test_img, "checkpoint/model2.ckpt")
if args.v:
    image_loader.plot_original_image(original_image, predicted, bbox)
