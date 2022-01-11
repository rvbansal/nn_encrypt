import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_binary_strings(n, string_len):
    return np.random.randint(0, 2, size=(n, string_len)) * 2 - 1


def init_net_weights(model, input_shape=None):
    if input_shape:
        model.build(input_shape)
    init_weights = []
    xavier_init = tf.keras.initializers.GlorotNormal()
    for weight in model.get_weights():
        init_weights.append(xavier_init(weight.shape))
    model.set_weights(init_weights)


def reconstruction_acc(msg, output):
    return tf.reduce_mean(
        tf.reduce_sum(tf.cast(tf.abs(msg - output) < 1, tf.float32), axis=1)
    )


def summarize_results(results, num_bits=32, num_steps=5000):
    successful_epochs = 0
    best_retrain_bob_accs = []
    best_retrain_eve_accs = []
    step_figs = []

    for epoch_result in results:
        (
            ab_step_accs,
            eve_step_accs,
            best_retrain_ab_acc,
            best_retrain_eve_acc,
        ) = epoch_result
        if best_retrain_ab_acc > 0:
            successful_epochs += 1
            best_retrain_bob_accs.append(best_retrain_ab_acc)
            best_retrain_eve_accs.append(best_retrain_eve_acc)
        step_figs.append(
            plot_accs(
                ab_step_accs, eve_step_accs, math.floor(num_bits / 2), num_bits, num_steps
            )
        )
    retrain_fig = plot_retrain_accs(best_retrain_bob_accs, best_retrain_eve_accs)

    print("Successful epochs: {} out of {}".format(successful_epochs, len(results)))
    for step_fig in step_figs:
        step_fig.show()
    retrain_fig.show()


def plot_accs(ab_step_accs, eve_step_accs, ymin=0, ymax=32, xmax=5000):
    fig = plt.figure()
    x = [i + 1 for i in range(len(ab_step_accs))]
    ab_accs = [i[1] for i in ab_step_accs]
    eve_accs = [i[2] for i in eve_step_accs]
    plt.plot(x, ab_accs, label="Bob")
    plt.plot(x, eve_accs, label="Eve")
    plt.xlabel("Training Step")
    plt.ylabel("Reconstruction Accuracy (in bits)")
    plt.legend()
    plt.ylim(ymin, ymax)
    plt.xlim(0, xmax)
    return fig


def plot_retrain_accs(best_retrain_bob_accs, best_retrain_eve_accs):
    fig = plt.figure()
    plt.scatter(best_retrain_eve_accs, best_retrain_bob_accs)
    plt.xlabel("Bob Reconstruction Accuracy (in bits)")
    plt.ylabel("Eve Reconstruction Accuracy (in bits)")
    return fig
