import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from dataset import loader
import random
FTRAIN = '/home/mihael/Documents/9. semestar/VIROKR/Projekt/Detecting-Facial-Features-CNN/dataset/kaggle/training.csv'

from dataset.face_tracking_dataset import load_dataset


np.random.seed(100)
tf.set_random_seed(100)
random.seed(100)
"""from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("MNIST_data/", one_hot=True)"""

X_train, y_train  = load_dataset(fname=FTRAIN)
# picture
pic_w = 96
pic_h = 96
num_classes = 15 * 2
# other
p_keep = tf.placeholder(tf.float32)
is_test = tf.placeholder(tf.bool)
iteration = tf.placeholder(tf.int32)
lr = tf.placeholder(tf.float32)
precision = tf.placeholder(dtype=tf.float32, shape=[None, 30])
ispravan_redak = tf.placeholder(dtype=tf.bool, shape=[None, 30])

# first con layer
n_filters1 = 24
# stride1 = [1,1,1,1]
filter1_w = 5
filter1_h = 5
# second con layer
n_filters2 = 36
stride2 = [1, 2, 2, 1]
filter2_w = 5
filter2_h = 5
# third con layer
n_filters3 = 48
# stride3 = [1,2,2,1]
filter3_w = 5
filter3_h = 5
# cetvrti
n_filters4 = 64
stride4 = [1, 2, 2, 1]
filter4_w = 3
filter4_h = 3
# pet
n_filters5 = 64
# first fc layer
n_neurons1 = 500
# second fc layer
n_neurons2 = 90
# third fc layer
n_neurons3 = num_classes

k = 1 * 1 * n_filters5

Wc1 = tf.Variable(tf.truncated_normal(shape=[filter1_w, filter1_h, 1, n_filters1], stddev=0.1, seed=100), name="wcon1")
Bc1 = tf.Variable(tf.zeros(shape=[n_filters1]), name="bcon1")
Wc2 = tf.Variable(tf.truncated_normal(shape=[filter2_w, filter2_h, n_filters1, n_filters2], stddev=0.1, seed=100), name="wcon2")
Bc2 = tf.Variable(tf.zeros(shape=[n_filters2]), name="bcon2")
Wc3 = tf.Variable(tf.truncated_normal(shape=[filter3_w, filter3_h, n_filters2, n_filters3], stddev=0.1, seed=100), name="wcon3")
Bc3 = tf.Variable(tf.zeros(shape=[n_filters3]), name="bcon3")
Wc4 = tf.Variable(tf.truncated_normal(shape=[filter4_w, filter4_h, n_filters3, n_filters4], stddev=0.1, seed=100), name="wcon4")
Bc4 = tf.Variable(tf.zeros(shape=[n_filters4]), name="bcon4")
Wc5 = tf.Variable(tf.truncated_normal(shape=[2, 2, n_filters4, n_filters5], stddev=0.1, seed=100), name="wcon5")
Bc5 = tf.Variable(tf.zeros(shape=[n_filters5]), name="bcon5")
Wf1 = tf.Variable(tf.truncated_normal(shape=[k, n_neurons1], stddev=0.1, seed=100), name="wf1")
Bf1 = tf.Variable(tf.zeros(shape=[n_neurons1]), name="bf1")
Wf2 = tf.Variable(tf.truncated_normal(shape=[n_neurons1, n_neurons2], stddev=0.1, seed=100), name="wf2")
Bf2 = tf.Variable(tf.zeros(shape=[n_neurons2]), name="bf2")
Wf3 = tf.Variable(tf.truncated_normal(shape=[n_neurons2, n_neurons3], stddev=0.1, seed=100), name="wf3")
Bf3 = tf.Variable(tf.zeros(shape=[n_neurons3]), name="bf3")


def newConLayer(input, W, B, pooling=True, activation_function=tf.nn.relu, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                index="", test_1=False):
    L = tf.nn.conv2d(input, W, strides=strides, padding='SAME') + B
 #   L = tf.Print(L, [tf.reduce_sum(L), tf.reduce_sum(W)
  #                   , tf.reduce_sum(B)], message="CONV TEST")
    convv = L
    if pooling:
        L = tf.nn.max_pool(value=L, ksize=ksize, strides=strides, padding='SAME')
        conll = L
    batch_norm, mua = batchnorm(L, is_test, iteration, convolutional=True)
    bnm = batch_norm
    A = activation_function(batch_norm)
    aa = A
    if test_1:
        return A, mua, [convv, conll, bnm, aa]
    return A, mua


def newFcLayer(input, W, B, activation_function=tf.nn.relu, last=False, index="", test1=False):
    if last:
        return tf.matmul(input, W) + B
    else:
        fc = tf.matmul(input, W) + B
     #   fc = tf.Print(fc, [tf.reduce_sum(W),
     #                                tf.reduce_sum(B),
     #                                tf.reduce_sum(fc)], message="FC TEST")
        batch_norm, mua = batchnorm(fc, is_test, iteration, convolutional=False)
        A = activation_function(batch_norm)
        dropout = tf.nn.dropout(A, keep_prob=p_keep, seed=100)

        if test1:
            return dropout, mua, [fc, batch_norm, A, dropout]
        return dropout, mua


def batchnorm(Ylogits, is_test, iteration, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, 0.0, 1.0, bnepsilon)
    return Ybn, update_moving_everages


X = tf.placeholder(tf.float32, shape=[None, pic_w * pic_h])
Y_labels = tf.placeholder(tf.float32, shape=[None, num_classes])
X_r = tf.reshape(X, shape=[-1, pic_w, pic_h, 1])
cl1, mau1, lss = newConLayer(X_r, Wc1, Bc1, pooling=True, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], test_1=True)
cl2, mau2 = newConLayer(cl1, Wc2, Bc2, pooling=True, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
cl3, mau3 = newConLayer(cl2, Wc3, Bc3, pooling=True, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
cl4, mau4 = newConLayer(cl3, Wc4, Bc4, pooling=True, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
cl5, mau5 = newConLayer(cl4, Wc5, Bc5, pooling=False)
flatten = tf.reshape(cl5, shape=[-1, k])
fc1, mau6, lss2 = newFcLayer(flatten, Wf1, Bf1, test1=True)
print(fc1 == lss2[-1])
fc2, mau7, lss3 = newFcLayer(fc1, Wf2, Bf2, test1=True)
fc3 = newFcLayer(fc2, Wf3, Bf3, last=True)
Y_predict = fc3

update = tf.group(mau1, mau2, mau3, mau4, mau5, mau6, mau7)

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=Y_labels)
# cross_entropy = tf.losses.absolute_difference(labels=Y_labels, predictions=Y_predict)
cross_entropy = tf.losses.mean_squared_error(labels=Y_labels, predictions=Y_predict)
#cross_entropy = tf.Print(cross_entropy, [cross_entropy, tf.reduce_sum(Y_labels),
#                                         tf.reduce_sum(Y_predict)])
# cross_entropy = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate = lr)
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
minimize = optimizer.minimize(cross_entropy)

"""predicted = tf.argmax(Y_predict, 1)
true = tf.argmax(Y_labels,1)
correct_predictions = tf.equal(predicted, true)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))"""
# accuracy = tf.metrics.accuracy(labels=Y_labels, predictions=Y_predict)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.less_equal(tf.abs(tf.subtract(Y_predict, Y_labels)), precision), ispravan_redak), tf.float32))
saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
train_batch_size = 60
test_batch_size = 60
num_iterations = 5000
prec = 0.05

gen = loader.next_batch(X_train, y_train, train_batch_size, 0)

for i in range(num_iterations):

    max_learning_rate = 0.01
    min_learning_rate = 0.0001
    s = 7500
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / (s * 0.145))

    x_train_batch, y_train_batch = next(gen)
    feed_dict_train = {X: x_train_batch, Y_labels: y_train_batch, lr: learning_rate, p_keep: 0.8, is_test: False,
                       iteration: i}
    """c1 = session.run(cl1,feed_dict=feed_dict_train)
    c2 = session.run(cl2,feed_dict=feed_dict_train)
    c3 = session.run(cl3,feed_dict=feed_dict_train)
    c4 = session.run(cl4,feed_dict=feed_dict_train)
    c5 = session.run(cl5,feed_dict=feed_dict_train)
    print(c1.shape)
    print(c2.shape)
    print(c3.shape)
    print(c4.shape)
    print(c5.shape)"""
    start = time.time()
   #  print("x_train_batch", np.sum(x_train_batch), "shape: ", x_train_batch.shape)
   #  for ly in lss:
   #      print("cl1: ly1", session.run(tf.reduce_sum(ly), feed_dict=feed_dict_train))
   #  print("cl1: ", session.run(tf.reduce_sum(cl1),feed_dict=feed_dict_train))
   #  print("cl2: ", session.run(tf.reduce_sum(cl2),feed_dict=feed_dict_train))
   #  print("cl3: ", session.run(tf.reduce_sum(cl3),feed_dict=feed_dict_train))
   #  print("cl4: ", session.run(tf.reduce_sum(cl4),feed_dict=feed_dict_train))
   #  print("cl5: ", session.run(tf.reduce_sum(cl5),feed_dict=feed_dict_train))
   #  print("flatten: ", session.run(tf.reduce_sum(flatten),feed_dict=feed_dict_train))
   # # for ly in lss2:
   # #     print("fc1: (ly)", session.run(tf.reduce_sum(ly), feed_dict=feed_dict_train))
   #  print("fc1: ", session.run(tf.reduce_sum(fc1),feed_dict=feed_dict_train))
   #  #print(fc1 == lss2[-1])
   #  for ly in lss3[:-1]:
   #      print("fc2: (ly)", session.run(tf.reduce_sum(ly), feed_dict=feed_dict_train))
   #  print("fc2: ", session.run(tf.reduce_sum(fc2),feed_dict=feed_dict_train))
   #  print("fc3: ", session.run(tf.reduce_sum(fc3),feed_dict=feed_dict_train))

    #print(session.run(tf.shape(cl1), feed_dict=feed_dict_train))
    #print(session.run([tf.shape(ll) for ll in lss], feed_dict_train))
    #print("=========")
    #print(session.run(tf.shape(cl2), feed_dict=feed_dict_train))
    #print(session.run(tf.shape(cl3), feed_dict=feed_dict_train))
    #print(session.run(tf.shape(cl4), feed_dict=feed_dict_train))
    ##print(session.run(tf.shape(cl5), feed_dict=feed_dict_train))
    #print(session.run(tf.shape(flatten), feed_dict=feed_dict_train))
    ##print(session.run(tf.shape(fc1), feed_dict=feed_dict_train))
    #print(session.run(tf.shape(fc2), feed_dict=feed_dict_train))
    #print(session.run(tf.shape(fc3), feed_dict=feed_dict_train))

    session.run(minimize, feed_dict=feed_dict_train)
    session.run(update, feed_dict=feed_dict_train)
  #  print(session.run(tf.reduce_sum(Y_predict), feed_dict=feed_dict_train))
    #print("Tensorflow operation took {:.2f} s".format(
    #    (time.time() - start)))  # print("iter: " + str(self.dataset_iter.get_iter_count())
    if (i % 100 == 0):
     #   x_train_batch, y_train_batch = loader.next_batch(X_train, y_train, test_batch_size, iteration)
        prec_tensor = np.array([[prec for j in range(30)] for i in range(test_batch_size)])
        ispr_red = np.array([[True for j in range(30)] for i in range(test_batch_size)])
        feed_dict_train = {X: x_train_batch, Y_labels: y_train_batch, lr: learning_rate, p_keep: 0.6, is_test: True,
                           iteration: i, precision: prec_tensor, ispravan_redak: ispr_red}
        loss, acc = session.run([cross_entropy, accuracy], feed_dict=feed_dict_train)
        print("iter:", i, "loss:", loss, "acc:", acc)

"""tf.add_to_collection('sir', Wc1)
tf.add_to_collection('sir', Bc1)        
tf.add_to_collection('sir', Wc2)        
tf.add_to_collection('sir', Bc2)        
tf.add_to_collection('sir', Wc3)        
tf.add_to_collection('sir', Bc3)        
tf.add_to_collection('sir', Wc4)        
tf.add_to_collection('sir', Bc4)        
tf.add_to_collection('sir', Wc5)        
tf.add_to_collection('sir', Bc5)
tf.add_to_collection('sir', Wf1)        
tf.add_to_collection('sir', Bf1)        
tf.add_to_collection('sir', Wf2)        
tf.add_to_collection('sir', Bf2)        
tf.add_to_collection('sir', Wf3)        
tf.add_to_collection('sir', Bf3)"""
save_path = saver.save(session, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)
"""x_test_batch=[]
y_test_batch = []
feed_dict_test = {X:x_test_batch, Y_labels:y_test_batch, lr:learning_rate, p_keep:1.0, is_test:True, iteration:4}
acc, p,last_layer = session.run([accuracy,predicted,fc3], feed_dict=feed_dict_test)
print("accuracy on 10000 test images -> {acc}".format(acc=acc))"""
"""img,lbl,pred,a_strenght = collect_wrong_predictions(x_test_batch,y_test_batch,p, last_layer)
plot_wrong_predictions(img, lbl,pred, a_strenght, bound=10)
plt.show()
img,lbl,pred,a_strenght = collect_right_predictions(x_test_batch,y_test_batch,p, last_layer)
plot_wrong_predictions(img, lbl,pred, a_strenght, bound=10)
plt.show()"""
