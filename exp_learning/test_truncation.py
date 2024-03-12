import tensorflow as tf


def truncation(x, m):
    ranking = tf.sort(tf.reshape(x, [-1]))

    # -1 because a value of 0.5 becomes 0 after rounding, so i want to set 0.5
    # the one immediately smaller than x_{m}, so that x_{m} will end up as 1
    # after rounding
    mu = ranking[-m - 1]

    sigmoid = 1.0 / (1.0 + tf.exp(-(x - mu)))
    return sigmoid


if __name__ == "__main__":
    x = tf.random.uniform([2, 3, 4])
    print(x)
    x = truncation(x, alpha=1.0, m=2)
    print(x)
    x = tf.math.round(x)
    print(x)
