#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

##================== PREPARE DATA ============================================##
sess = tf.InteractiveSession()
X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

def enlarge_random_loc(x):
    """ Enlarge the image to 40x40, and distort it.

    Examples
    ---------
    x = enlarge_random_loc(X_train[0])
    print(x, x.shape, x.max())
    tl.vis.save_image(x, '_xf.png')
    tl.vis.save_image(X_train[0], '_x.png')
    """
    b = np.zeros((40, 40, 1))
    o = int((40-28)/2)
    b[o:o+28, o:o+28] = x
    x = b
    x = tl.prepro.rotation(x, rg=30, is_random=True, fill_mode='constant')
    x = tl.prepro.shear(x, 0.05, is_random=True, fill_mode='constant')
    x = tl.prepro.shift(x, wrg=0.30, hrg=0.30, is_random=True, fill_mode='constant')
    x = tl.prepro.zoom(x, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    return x

def to_large_imgs(X, y):
    """ Enlarge images to 40x40, and distort them. """
    X_40 = []
    for X_a, y_a in tl.iterate.minibatches(X, y, 50, shuffle=False):
        X_40.extend(tl.prepro.threading_data(X_a, fn=enlarge_random_loc))
    X_40 = np.asarray(X_40)
    return X_40

X_train_40 = to_large_imgs(X_train, y_train)
X_val_40 = to_large_imgs(X_val, y_val)
X_test_40 = to_large_imgs(X_test, y_test)

tl.vis.save_images(X_train[0:64], [8, 8], '_distort_imgs_original.png')
tl.vis.save_images(X_train_40[0:64], [8, 8], '_distort_imgs.png')

##================== DEFINE MODEL ============================================##
batch_size = 64
# create dataset with size of 40x40 and random location
x = tf.placeholder(tf.float32, shape=[batch_size, 40, 40, 1], name='x')
y_ = tf.placeholder(tf.int64, shape=[batch_size, ], name='y_')

def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        nin = InputLayer(x, name='in')
        ## 1. Localisation network
        # use MLP as the localisation net
        nt = FlattenLayer(nin, name='tf')
        nt = DenseLayer(nt, n_units=20, act=tf.nn.tanh, name='td1')
        nt = DropoutLayer(nt, 0.8, True, is_train, name='tdrop')
        # you can also use CNN instead for MLP as the Localisation net
        # nt = Conv2d(nin, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc1')
        # nt = Conv2d(nt, 8, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')
        ## 2. Spatial transformer module (sampler)
        n = SpatialTransformer2dAffineLayer(nin, nt, out_size=[40, 40], name='stn')
        s = n
        ## 3. Classificer
        n = Conv2d(n, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='c1')
        n = Conv2d(n, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='c2')
        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=tf.nn.relu, name='d1')
        n = DenseLayer(n, n_units=10, act=tf.identity, name='do')
        return n, s

net_train, _ = model(x, is_train=True, reuse=False)
net_test, net_trans = model(x, is_train=False, reuse=True)

##================== DEFINE TRAIN OPS ========================================##
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, 'cost')

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

n_epoch = 200
learning_rate = 0.0001
print_freq = 10

train_params = tl.layers.get_variables_with_name('STN', train_only=True, printable=True)
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

##================== TRAINING ================================================##
tl.layers.initialize_global_variables(sess)
net_train.print_params()
net_train.print_layers()

for epoch in range(n_epoch):
    start_time = time.time()
    ## you can also try continuous data augmentation
    # for X_train_a, y_train_a in tl.iterate.minibatches(
    #                             X_train, y_train, batch_size, shuffle=True):
    #     X_train_a = tl.prepro.threading_data(X_train_a, fn=enlarge_random_loc)
    for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train_40, y_train, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train_40, y_train, batch_size, shuffle=False):
            err, ac = sess.run([cost, acc], feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(
                                    X_val_40, y_val, batch_size, shuffle=False):
            err, ac = sess.run([cost, acc], feed_dict={x: X_train_a, y_: y_train_a})
            val_loss += err; val_acc += ac; n_batch += 1
        print("   val loss: %f" % (val_loss/ n_batch))
        print("   val acc: %f" % (val_acc/ n_batch))

##================== EVALUATION ==============================================##
print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(
                            X_test_40, y_test, batch_size, shuffle=False):
    err, ac = sess.run([cost, acc], feed_dict={x: X_test_a, y_: y_test_a})
    test_loss += err; test_acc += ac; n_batch += 1
print("   test loss: %f" % (test_loss/n_batch))
print("   test acc: %f" % (test_acc/n_batch))

print('Save images')
trans_imgs = sess.run(net_trans.outputs, {x: X_test_40[0:64]})
tl.vis.save_images(X_test_40[0:64], [8, 8], '_before_stn.png')
tl.vis.save_images(trans_imgs, [8, 8], '_after_stn.png')
