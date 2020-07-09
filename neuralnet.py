import numpy as np
import sys
import string
import tensorflow as tf


class Model:

    def __init__(self, sess, name, layer, learning_rate=0.001, keep_prob=0.8):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self.dropout_rate = keep_prob
        self.layer = layer
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, [None, self.layer[0]])
        self.Y = tf.placeholder(tf.int32, [None, 1])

        Y_one_hot = tf.one_hot(self.Y, self.layer[len(self.layer) - 1])
        Y_one_hot = tf.reshape(Y_one_hot, [-1, self.layer[len(self.layer) - 1]])

        self.keep_prob = tf.placeholder(tf.float32)

        # weights & biases for nn layers
        W1 = tf.get_variable("W1", shape=[self.layer[0], self.layer[1]],
                             initializer=tf.keras.initializers.he_normal())
        b1 = tf.Variable(tf.random_normal([self.layer[1]]))
        L1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

        W2 = tf.get_variable("W2", shape=[self.layer[1], self.layer[2]],
                             initializer=tf.keras.initializers.he_normal())
        b2 = tf.Variable(tf.random_normal([self.layer[2]]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

        W3 = tf.get_variable("W3", shape=[self.layer[2], self.layer[3]],
                             initializer=tf.keras.initializers.he_normal())
        b3 = tf.Variable(tf.random_normal([self.layer[3]]))
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
        L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

        W4 = tf.get_variable("W4", shape=[self.layer[3], self.layer[4]],
                             initializer=tf.keras.initializers.he_normal())
        b4 = tf.Variable(tf.random_normal([self.layer[4]]))
        self.hypothesis = tf.matmul(L3, W4) + b4
        self.softmax_res = tf.nn.softmax(logits=self.hypothesis)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_one_hot, logits=self.hypothesis))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        prediction = tf.argmax(self.hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver(tf.global_variables())

    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.softmax_res,
                             feed_dict={self.X: x_test, self.keep_prob: keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: self.dropout_rate})


def normalize_data(x, data_mean, data_std):
    data = x
    data[:, :-1] -= data_mean[None, :-1]
    data[:, :-1] /= data_std[None, :-1]

    return data


def get_minibatch(x_train, y_train, batch_size):
    data_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data_set = data_set.shuffle(np.shape(x_train)[0]).repeat().batch(batch_size)
    iterator = data_set.make_one_shot_iterator()
    x, y = iterator.get_next()

    return x, y


def main():

    text_path = 'dev_text.txt'
    label_path = 'dev_label.txt'
    heldout_text_path = 'heldout_text.txt'
    heldout_label_path = 'heldout_pred_nn.txt'
    with open(text_path) as text_file:
        texts = text_file.read().split('\n')
    with open(label_path) as label_file:
        labels = label_file.read().split('\n')

    ratio = 1.0
    feat_dim = 500
    train_texts = texts[:int(ratio*len(texts))]
    test_texts = texts[int(ratio*len(texts)):]
    train_labels = labels[:int(ratio*len(labels))]
    test_labels = labels[int(ratio*len(labels)):]

    # ----------------- tokenization -----------------

    vocab = {}
    for line in train_texts:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.lower()
        tokens = line.split()
        for token in tokens:
            contain_digit = any(map(str.isdigit, token))
            if contain_digit:
                continue
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
    with open('stopword.list') as stopword_file:
        stopword_list = stopword_file.read().split('\n')
    for stopword in stopword_list:
        vocab.pop(stopword, None)
    vocab.pop('br')
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda vocab: vocab[1], reverse=True)}

    # ----------------- feature extraction -----------------

    # vocab_list = ['OOV'] + list(vocab)[:1999]
    vocab_list = list(vocab)[:feat_dim]
    vocab_top500 = {vocab_list[i]: i for i in range(0, len(vocab_list))}

    CTF_features = []
    for line in train_texts:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.lower()
        tokens = line.split()
        CTF_feature = np.zeros(feat_dim)
        for token in tokens:
            contain_digit = any(map(str.isdigit, token))
            if contain_digit:
                continue
            if token in vocab_list:
                CTF_feature[vocab_top500[token]] += 1
            # if token not in vocab_list:
            #     CTF_feature[0] += 1
            # else:
            #     CTF_feature[vocab_top500[token]] += 1
        CTF_features.append(CTF_feature)
    CTF_features = np.asarray(CTF_features)
    train_x = CTF_features

    with open(heldout_text_path) as heldout_text_file:
        heldout_texts = heldout_text_file.read().split('\n')

    CTF_features = []
    for line in heldout_texts:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.lower()
        tokens = line.split()
        CTF_feature = np.zeros(feat_dim)
        for token in tokens:
            contain_digit = any(map(str.isdigit, token))
            if contain_digit:
                continue
            if token in vocab_list:
                CTF_feature[vocab_top500[token]] += 1
            # if token not in vocab_list:
            #     CTF_feature[0] += 1
            # else:
            #     CTF_feature[vocab_top500[token]] += 1
        CTF_features.append(CTF_feature)
    CTF_features = np.asarray(CTF_features)

    for i in range(len(train_labels)):
        if train_labels[i] == 'pos':
            train_labels[i] = 0
        elif train_labels[i] == 'neg':
            train_labels[i] = 1
    train_labels = np.asarray(train_labels)
    train_labels = np.reshape(train_labels, (-1, 1))

    layer = [feat_dim, 1000, 500, 200, 2]
    learning_rate = 0.001
    dropout_rate = 0.8
    mini_batch_size = 256
    epochs = 3000
    # config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session() as sess:
        m = Model(sess, "m", layer, learning_rate, dropout_rate)

        sess.run(tf.global_variables_initializer())
        early_stop = 0

        x, y = get_minibatch(train_x, train_labels, mini_batch_size)

        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(np.size(train_labels) / mini_batch_size)

            for ii in range(total_batch):
                x_batch, y_batch = sess.run([x, y])

                c, _ = m.train(x_batch, y_batch)
                avg_cost += c / total_batch
            print("Epoch {} Completed".format(epoch))

        predict = m.predict(CTF_features)
        prediction = np.argmax(predict, axis=1)

        result = ['neg' for i in range(CTF_features.shape[0])]
        for i in range(CTF_features.shape[0]):
            if prediction[i] == 0:
                result[i] = 'pos'
        result = np.asarray(result)
        np.savetxt(heldout_label_path, result, fmt='%s')

        print("test")

    print("stop")

    return


if __name__ == '__main__':
    main()
