import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv1D
from tensorflow.keras.activations import relu, tanh


class ConvolutionalsNet(Model):
    def __init__(self, num_conv_layers, output_dims, kernel_sizes, strides, input_dim):
        super().__init__()
        assert len(output_dims) == len(strides) == len(kernel_sizes) == num_conv_layers
        self.num_conv_layers = num_conv_layers
        self.output_dims = output_dims
        self.input_dims = [input_dim] + self.output_dims[-1:]
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.convolutions = self.init_convolutions()

    def init_convolutions(self):
        conv_layers = tf.keras.Sequential()
        for l in range(self.num_conv_layers):
            conv1d = Conv1D(
                self.output_dims[l],
                self.kernel_sizes[l],
                strides=self.strides[l],
                padding="same",
            )
            conv_layers.add(conv1d)
            activation = (
                Activation(tanh) if l == self.num_conv_layers - 1 else Activation(relu)
            )
            conv_layers.add(activation)
        return conv_layers

    def call(self, x):
        return self.convolutions(x)
