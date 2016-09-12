import tensorflow as tf
from abc import ABCMeta, abstractmethod

class RnnModel(metaclass=ABCMeta):
    def __init__(self, learning_rate, batch_size, n_out, n_channels, samps_per_step):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_out = n_out
        self.n_channels = n_channels
        self.samps_per_step = samps_per_step

        self.x = tf.placeholder("float", [None, self.n_steps, self.samps_per_step])
        self.y = tf.placeholder("float", [None, self.n_classes])

        self.weights, self.biases = gen_wb()

        with tf.name_scope('Model'):
            self.pred = gen_model(self.x, self.weights, self.biases)

        # Define loss and optimizer
        with tf.name_scope('Loss'):
            self.cost = gen_cost()

        with tf.name_scope('SGD'):
            optimizer = gen_optimizer()

        # Evaluate model
        with tf.name_scope('Accuracy'):
            self.accuracy = gen_accuracy()

    def get_optimizer(self):
        return optimizer

    def get_pred(self):
        return pred

    def get_cost(self):
        return cost

    def get_accuracy(self):
        return accuracy

    @abstractmethod
    def gen_model(self, x, weights, biases):
        pass

    @abstractmethod
    def gen_cost(self):
        pass

    @abstractmethod
    def gen_accuracy(self):
        pass

    @abstractmethod
    def gen_optimizer(self):
        pass

    @abstractmethod
    def gen_wb(self):
        pass

class BiRNN(RnnModel):

    def gen_model(self,x, weights, biases):
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, self.samps_per_step])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, self.n_steps, x)

        fwd_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
        bwd_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
        outputs, _, _ = tf.nn.bidirectional_rnn(fwd_cell, bwd_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def gen_cost(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))

    def gen_accuracy(self):
        accuracy = tf.equal(tf.argmax(self.pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
        return accuracy

    def gen_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    def gen_wb(self):
        weights = {
            'out': tf.Variable(tf.random_normal([2*self.n_hidden, self.n_classes]))
            }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
            }
        return weights, biases


class RNN(RnnModel):

    def gen_model(self,x, weights, biases):
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, self.samps_per_step])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, self.n_steps, x)

        cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
        outputs, _, _ = tf.nn.bidirectional_rnn(cell x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def gen_cost(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))

    def gen_accuracy(self):
        accuracy = tf.equal(tf.argmax(self.pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
        return accuracy

    def gen_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    def gen_wb(self):
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
            }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
            }
        return weights, biases


