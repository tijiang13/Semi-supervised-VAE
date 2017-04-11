import cPickle
import gzip

import numpy as np
import random

MNIST_PATH = './data/mnist_28.pkl.gz'

'''
    the following 3 functions came from the original implementation by D.P. Kingma
        @https://github.com/dpkingma/nips14-ssl
    these function are slightly modified for convenience
'''
def load_mnist(path):
    f = gzip.open(path, 'rb')
    train, valid, test = cPickle.load(f)
    f.close()
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x,  test_y  = test
    return train_x, train_y, valid_x, valid_y, test_x, test_y

# Loads data where data is split into class labels
def load_mnist_split(path = MNIST_PATH):
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(path)

    def split_by_class(x, y, num_classes):
        result_x = [0]*num_classes
        result_y = [0]*num_classes
        for i in range(num_classes):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y

    train_x, train_y = split_by_class(train_x, train_y, 10)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def create_semisupervised(x, y, n_labeled):
    n_x = x[0].shape[0]
    n_classes = 10
    if n_labeled % n_classes != 0:
        raise("n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
    n_labels_per_class = n_labeled//n_classes
    x_labeled = [0]*n_classes
    x_unlabeled = [0]*n_classes
    y_labeled = [0]*n_classes
    y_unlabeled = [0]*n_classes
    for i in range(n_classes):
        idx = range(x[i].shape[0])
        random.shuffle(idx)
        x_labeled[i]   = x[i][idx[:n_labels_per_class]]
        y_labeled[i]   = y[i][idx[:n_labels_per_class]]
        x_unlabeled[i] = x[i][idx[n_labels_per_class:]]
        y_unlabeled[i] = y[i][idx[n_labels_per_class:]]
    return np.vstack(x_labeled), np.hstack(y_labeled), np.vstack(x_unlabeled), np.hstack(y_unlabeled)

def batch_generator(data, batch_size, num_epoch, shuffle = True):
    data = list(data)
    data = np.array(data)
    data_size = data.shape[0]
    num_batches_per_epoch = (data_size + batch_size - 1)//batch_size
    for epoch in range(num_epoch):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_idx in range(num_batches_per_epoch):
            start_idx = batch_idx * batch_size
            end_idx   = min((batch_idx + 1)*batch_size, data_size)
            yield(shuffled_data[start_idx:end_idx])

def ssl_batch_gen(x_l, y_l, x_u, batch_size, n_epoch, shuffle = True):
    '''
        x_l, y_l: x & its labelled
        x_u: x without labels
        batch_size: num of samples in a single batch
        epoch: the # of epochs that go through the full dataset

        Note:
            1. x_l, y_l & x_u are supposed to be numpy array
    '''
    # decide # of label data in a single batch
    num_labelled   = x_l.shape[0]
    num_unlabelled = x_u.shape[0]
    num_data       = num_labelled + num_unlabelled

    batch_per_epoch      = num_data       // batch_size
    labelled_per_batch   = num_labelled   // batch_per_epoch
    unlabelled_per_batch = num_unlabelled // batch_per_epoch

    assert(unlabelled_per_batch == batch_size - labelled_per_batch)

    gen_labelled   = batch_generator(zip(x_l, y_l), labelled_per_batch, n_epoch, shuffle)
    gen_unlabelled = batch_generator(zip(x_u), unlabelled_per_batch, n_epoch, shuffle)
    return gen_labelled, gen_unlabelled

def get_dims(data, dim_id):
    return data[:, dim_id]

def sample(mu, logvar):
    n_samples = mu.shape[0]
    n_dim = mu.shape[1]
    eps = np.random.normal(size = (n_samples, n_dim))
    std = np.exp(logvar * 0.5)
    return mu + eps * std

import time
def time_since(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= 60 * m

    return "%d m %d s" % (m, s)
