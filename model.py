import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.metrics import mae
from tensorflow.keras.optimizers import Adam

from convolutionals_net import ConvolutionalsNet
from tools import generate_binary_strings, init_net_weights, reconstruction_acc


class CommunicateNet(Model):
    def __init__(
        self, msg_len, key_len, num_conv_layers, output_dims, kernel_sizes, strides
    ):
        super(CommunicateNet, self).__init__()
        self.dense = Dense(msg_len + key_len)
        self.sigmoid = Activation(sigmoid)
        self.convolutions = ConvolutionalsNet(
            num_conv_layers, output_dims, kernel_sizes, strides, msg_len + key_len
        )

    def call(self, x):
        x = self.dense(x)
        x = self.sigmoid(x)
        x = tf.expand_dims(x, axis=2)
        return tf.squeeze(self.convolutions(x), axis=2)


class EveNet(Model):
    def __init__(
        self, msg_len, key_len, num_conv_layers, output_dims, kernel_sizes, strides
    ):
        super(EveNet, self).__init__()
        self.dense1 = Dense(msg_len + key_len)
        self.dense2 = Dense(msg_len + key_len)
        self.sigmoid1 = Activation(sigmoid)
        self.sigmoid2 = Activation(sigmoid)
        self.convolutions = ConvolutionalsNet(
            num_conv_layers, output_dims, kernel_sizes, strides, msg_len + key_len
        )

    def call(self, x):
        x = self.dense1(x)
        x = self.sigmoid1(x)
        x = self.dense2(x)
        x = self.sigmoid2(x)
        x = tf.expand_dims(x, axis=2)
        return tf.squeeze(self.convolutions(x), axis=2)


class EncryptNet:
    def __init__(
        self,
        net_params,
        lr,
        batch_size,
        epochs,
        iterations,
        eve_retrain_bool,
        eve_retrain_epochs,
        eve_retrain_iterations,
        eval_acc_every=100,
    ):
        self.net_params = net_params
        self.msg_len = self.net_params["msg_len"]
        self.key_len = self.net_params["key_len"]
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = iterations
        self.ab_acc_threshold = self.msg_len - 1
        self.eve_acc_threshold = int(self.msg_len * 0.5) + 2
        self.eve_retrain_bool = eve_retrain_bool
        self.eve_retrain_epochs = eve_retrain_epochs
        self.eve_retrain_iterations = eve_retrain_iterations
        self.eval_acc_every = eval_acc_every
        self._init_nets()

    def _init_nets(self):
        self.alice = CommunicateNet(**self.net_params)
        self.bob = CommunicateNet(**self.net_params)
        self.eve = EveNet(**self.net_params)

    def train(self):
        all_results = []
        for e in range(self.epochs):
            print("Training epoch {}".format(e + 1))
            init_net_weights(self.alice, (1, self.msg_len + self.key_len))
            init_net_weights(self.bob, (1, 2 * self.key_len))
            init_net_weights(self.eve, (1, self.msg_len))
            all_results.append(self._train_epoch())
        return all_results

    def _train_epoch(self):
        ab_weights = self.alice.trainable_variables + self.bob.trainable_variables
        eve_weights = self.eve.trainable_variables
        ab_opt, eve_opt = Adam(self.lr), Adam(self.lr)

        ab_results, eve_results = [], []
        for i in range(self.iterations):
            ab_iter_results = self._train_step("ab", ab_opt, ab_weights)
            eve_iter_results = self._train_step("eve", eve_opt, eve_weights)
            ab_results.append(ab_iter_results)
            eve_results.append(eve_iter_results)

            if (i + 1) % self.eval_acc_every == 0:
                print("Step: {}".format(i + 1))
                print("Bob Accuracy: {}".format(ab_iter_results[1]))
                print("Eve Accuracy: {}".format(eve_iter_results[2]))

            if (
                self.eve_retrain_bool
                and ab_iter_results[1] >= self.ab_acc_threshold
                and ab_iter_results[2] <= self.eve_acc_threshold
            ):
                print("Successful epoch")
                print("Step: {}".format(i + 1))
                print("Bob Accuracy: {}".format(ab_iter_results[1]))
                print("Eve Accuracy: {}".format(eve_iter_results[2]))
                
                best_retrain_ab_acc, best_retrain_eve_acc = self._retrain_eve()
                return (
                    ab_results,
                    eve_results,
                    best_retrain_ab_acc,
                    best_retrain_eve_acc,
                )
        
        if self.eve_retrain_bool:
            print("Unsuccessful epoch")
        
        return ab_results, eve_results, float("-inf"), float("inf")

    def _train_step(self, network, optimizer, weights):
        batch_size = self.batch_size if network == "ab" else 2 * self.batch_size
        msg = generate_binary_strings(batch_size, self.msg_len)
        key = generate_binary_strings(batch_size, self.key_len)
        alice_input = tf.concat([msg, key], axis=1)

        with tf.GradientTape() as tape:
            alice_output = self.alice(alice_input, training=True)
            bob_output = self.bob(tf.concat([alice_output, key], axis=1))
            eve_output = self.eve(alice_output)
            bob_err = tf.reduce_mean(mae(msg, bob_output))
            eve_err = tf.reduce_mean(mae(msg, eve_output))
            ab_loss = bob_err + (1 - eve_err) ** 2
            network_loss = ab_loss if network == "ab" else eve_err
        gradients = tape.gradient(network_loss, weights)
        optimizer.apply_gradients(zip(gradients, weights))

        bob_acc = reconstruction_acc(msg, bob_output)
        eve_acc = reconstruction_acc(msg, eve_output)

        return float(network_loss), float(bob_acc), float(eve_acc)

    def _retrain_eve(self):
        eve_opt = Adam(self.lr)
        eve_weights = self.eve.trainable_variables
        best_retrain_eve_acc = float("-inf")
        best_retrain_ab_acc = float("-inf")

        for e in range(self.eve_retrain_epochs):
            print("Retraining Eve epoch {}".format(e + 1))
            init_net_weights(self.eve)
            for _ in range(self.eve_retrain_iterations):
                eve_iter_results = self._train_step("eve", eve_opt, eve_weights)
                if eve_iter_results[2] >= best_retrain_eve_acc:
                    best_retrain_eve_acc = eve_iter_results[2]
                    best_retrain_ab_acc = eve_iter_results[1]

        return best_retrain_ab_acc, best_retrain_eve_acc
