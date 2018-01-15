import tensorflow as tf
import time

from batch.dataset_iterator import DatasetIterator
from batch.std_data_iterator import StandardDatasetIterator
from fetches.i_optimizer import Optimizer
from fetches.i_loss import Loss
from net.layers.i_layer import Layer
from net.layers.input_layer import InputLayer
from placeholders.i_placeholder import Placeholder
from placeholders.data_placeholder import DataPlaceholder


class Network:
    def __init__(self, dataset_iter: DatasetIterator = None, print_fn=print):
        self.layers = []
        self.updates = []
        self.built = False
        self.placeholders = set()
        self.metrics_placeholders = set()
        self.feed_dict = {}
        self.metrics_dict = {}
        self.fetches = []
        self.metrics_fetches = []
        self.curr_iter = 0
        self.is_test = None
        self.dataset_iter = dataset_iter if dataset_iter is not None else StandardDatasetIterator()
        self.sess = None
        self.print_fn = print_fn

    def add_layer(self, layer: Layer):
        assert self.built != True
        self.layers.append(layer)
        return self

    def build(self, input: DataPlaceholder, output: DataPlaceholder,
              optimizer: Optimizer, loss: Loss, metrics: list):
        self.input = input
        self.input.set_data_iterator(self.dataset_iter)
        self.output = output
        #  self.debug_list = []
        # self.debug_list_names = []
        self.output.set_data_iterator(self.dataset_iter)
        if len(self.layers) < 1:
            raise ValueError("At least one layer was expected!")
        first = InputLayer(input.get_shape(), "input")
        first.initialize(input.get_forward_value(), self)
        self.layers[0].initialize(first, self)
        # self.debug_list_names.append(type(self.layers[0]))
        # self.debug_list.append(self.layers[0].output)
        if self.layers[0].has_update_values():
            self.updates.append(self.layers[0].get_update_values())
        for i in range(1, len(self.layers)):
            self.layers[i].initialize(self.layers[i - 1], self)
            # if type(self.layers[i]) in [FlattenLayer, ActivationLayer,
            #                             DropoutLayer]:
            #            self.debug_list_names.append(type(self.layers[i]))
            #            self.debug_list.append(self.layers[i].output)
            if self.layers[i].has_update_values():
                self.updates.append(self.layers[i].get_update_values())
        self.net = self.layers[-1].output
        # self.debug_list.append(self.net)
        # self.debug_list_names.append(type(self.layers[-1]))

        self.add_placeholder(input)
        self.add_placeholder(output)

        self.update_fetch = tf.group(*self.updates)

        self.metrics = [loss] + metrics

        self.optimizer = optimizer
        self.loss = loss
        self.loss.initialize(self)
        self.optimizer.initialize(self)
        self.fetches.append(optimizer.to_fetch())
        self.metrics_fetches.append(loss.to_fetch())
        for m in metrics:
            m.initialize(self)
            self.metrics_fetches.append(m.to_fetch())
            m_pholders = m.get_placeholders()
            if m_pholders is not None:
                for p in m_pholders:
                    self.metrics_dict[p.placeholder] = p.get_update_value()
        self.metrics_placeholders.update(self.placeholders)
        self.built = True

    def update_metrics(self, values):
        self.print_fn("epoch " + str(self.dataset_iter.get_epoch_count()) + ", total iter " + str(
            self.dataset_iter.get_iter_count()) + ": ")
        for m, v in zip(self.metrics, values):
            self.print_fn(str("   " + m.print_status(v)))

    def add_placeholder(self, placeholder: Placeholder):
        self.placeholders.add(placeholder)

    def remove_placeholder(self, placeholder: Placeholder):
        self.placeholders.remove(placeholder)

    def add_metrics_placeholder(self, placeholder: Placeholder):
        self.placeholders.add(placeholder)

    def remove_metrics_placeholder(self, placeholder: Placeholder):
        self.metrics_placeholders.add(placeholder)

    def update_metrics_dict(self):
        for p in self.metrics_placeholders:
            self.metrics_dict[p.placeholder] = p.get_update_value()

    def update_feed_dict(self):
        for p in self.placeholders:
            self.feed_dict[p.placeholder] = p.get_update_value()

    def train(self, X, y, num_epochs=30, print_step=100, save_file=None,
              load_file=None, writer_file=None):
        # summaries
        tf.summary.scalar('loss', self.loss.to_fetch())
        tf.summary.scalar('accuracy', self.metrics[1].to_fetch())
        writer = tf.summary.FileWriter(writer_file)
        assert self.built
        self.is_test = False
        if save_file is not None or load_file is not None:
            saver = tf.train.Saver()
        self.sess = tf.Session()
        writer.add_graph(self.sess.graph)
        if load_file is not None:
            saver.restore(self.sess, load_file)
        else:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
        for _ in self.dataset_iter.iterate(X, y, num_epochs):
            self.update_feed_dict()
            start = time.time()
            # print("x_train_batch", np.sum(self.dataset_iter.get_current_batch_X()),
            #       ", shape:", self.dataset_iter.get_current_batch_X().shape)
            # for obj, out in zip(self.debug_list_names, self.debug_list):
            #     print(obj, self.sess.run(tf.reduce_sum(out), self.feed_dict))
            self.sess.run(self.fetches, self.feed_dict)
            self.sess.run(self.update_fetch, self.feed_dict)
            # print(self.sess.run(tf.reduce_sum(self.net), feed_dict=self.feed_dict))
            #  print("Tensorflow operation took {:.2f} s".format((time.time() - start)))          #  print("iter: " + str(self.dataset_iter.get_iter_count())
            #        + ", epoch=" + str(self.dataset_iter.get_epoch_count())
            #        + ", batch=" + str(self.dataset_iter.get_batch_count()))
            if self.dataset_iter.get_iter_count() % print_step == 0:
                self.is_test = True
                self.update_metrics_dict()
                self.add_summary(writer)
                self.update_metrics(self.sess.run(self.metrics_fetches, self.metrics_dict))
                self.is_test = False
        if save_file is not None:
            saver.save(self.sess, save_file)

    def close_session(self):
        self.sess.close()

    def evaluate(self, X, y, load_file=None, writer_file=None):
        # summaries
        tf.summary.scalar('loss', self.loss.to_fetch())
        tf.summary.scalar('accuracy', self.metrics[1].to_fetch())
        writer = tf.summary.FileWriter(writer_file)
        assert self.built
        if self.sess is None:
            self.sess = tf.Session()
        writer.add_graph(self.sess.graph)
        self.is_test = True
        if load_file is not None:
            saver = tf.train.Saver()
            if self.sess is not None:
                self.sess.close()
            saver.restore(self.sess, load_file)

        for _ in self.dataset_iter.iterate(X, y, 10):
            self.update_feed_dict()
            self.update_metrics_dict()
            self.add_summary(writer)
            self.update_metrics(self.sess.run(self.metrics_fetches, self.metrics_dict))

    def predict(self, x, load_file=None):
        assert self.built
        self.is_test = True
        if self.sess is None:
            self.sess = tf.Session()
        if load_file is not None:
            saver = tf.train.Saver()
            if self.sess is not None:
                self.sess.close()
            saver.restore(self.sess, load_file)
        next(self.dataset_iter.iterate(x))
        self.update_feed_dict()
        self.feed_dict.pop(self.output.placeholder)
        self.feed_dict[self.input.placeholder] = x
        predicted = self.sess.run(self.net, self.feed_dict)
        return predicted


        # def test_train(self, X_train, y_train):
        #     self.sess = tf.Session()
        #     self.sess.run(tf.global_variables_initializer())
        #
        #     iter = IterationCounter(self.dataset_iter)
        #     is_test = IsTestPlaceholder(self)
        #     for i in range(5000):
        #
        #         max_learning_rate = 0.01
        #         min_learning_rate = 0.0001
        #         s = 7500
        #         learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i / (s * 0.145))
        #
        #         x_train_batch, y_train_batch = self.test_next_batch(X_train, y_train, 60)
        #         feed_dict_train = {self.input.placeholder: x_train_batch, self.output.placeholder: y_train_batch,
        #                            self.optimizer.lr.placeholder: learning_rate, is_test.placeholder: False,
        #                            iter.placeholder: i}
        #         start = time.time()
        #         print([v for v in tf.trainable_variables()])
        #         self.sess.run(self.optimizer.to_fetch(), feed_dict=feed_dict_train)
        #         self.sess.run(self.updates, feed_dict=feed_dict_train)
        #         print("Tensorflow operation took {:.2f} s".format(
        #             (time.time() - start)))

        # def test_next_batch(self, X, y,  size):
        #     index = np.random.randint(0, X.shape[0] - size)
        #     return X[index:index + size], y[index:index + size]

    def add_summary(self, writer):
        merged = tf.summary.merge_all()
        summary = self.sess.run(merged, self.metrics_dict)
        writer.add_summary(summary, self.dataset_iter.get_iter_count())
