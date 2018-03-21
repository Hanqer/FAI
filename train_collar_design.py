import os
from random import shuffle
import random
from PIL import Image
import time
import numpy as np
import csv
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import (inception_v3, inception_v3_arg_scope)

slim = tf.contrib.slim

model_dir = "model/model_collar_design"
logs_dir = "./logs/collar_design"
label_dir = './Annotations/collar_design_labels.csv'
print_dir = './logs/logs_collar_design.txt'

def load_image(path):
    # load image
    #img = skimage.io.imread(path)
    img = Image.open(path)
    if random.random() < 0.3:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img)
    img = img / 255.0
    if ((0 <= img).all() and (img <= 1.0).all()) is False:
        raise Exception("image value should be [0, 1]")
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    # resize to 299, 299
    resized_img = skimage.transform.resize(crop_img, (299, 299))
    return resized_img

# with tf.device('/cpu:0'):
x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
real_y = tf.placeholder(tf.int64, [None, ], name='real_y')
#isTrain = tf.placeholder(tf.bool, shape=[1])

net_in = tl.layers.InputLayer(x, name='input_layer')
with slim.arg_scope(inception_v3_arg_scope()):
    ## Alternatively, you should implement inception_v3 without TensorLayer as follow.
    # logits, end_points = inception_v3(X, num_classes=1001,
    #                                   is_training=False)
    network = tl.layers.SlimNetsLayer(
        prev_layer=net_in,
        slim_layer=inception_v3,
        slim_args={
            'num_classes': 5,
            'is_training': True,
            'dropout_keep_prob' : 0.8,       # for training
            #  'min_depth' : 16,
            #  'depth_multiplier' : 1.0,
            #  'prediction_fn' : slim.softmax,
            #  'spatial_squeeze' : True,
            #  'reuse' : None,
            #  'scope' : 'InceptionV3'
        },
        name='InceptionV3'  # <-- the name should be the same with the ckpt model
    )

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
network.print_params(False)


y = network.outputs
probs = tf.nn.softmax(y)
loss = tl.cost.cross_entropy(y, real_y, 'myloss')
tf.summary.scalar('loss', loss)

optim = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(loss)
probs = tf.nn.softmax(y)
correct = tf.equal(tf.argmax(y, 1), real_y)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar('acc', acc)



outlogs = open(print_dir, 'w')

tl.files.exists_or_mkdir(model_dir)
tl.files.exists_or_mkdir(logs_dir)

merged = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter(logs_dir+'/train', sess.graph)
#val_summary_writer = tf.summary.FileWriter(logs_dir+'/val')

tl.layers.initialize_global_variables(sess)
saver = tf.train.Saver()
saver2 = tf.train.Saver(network.all_params[0:384])
saver2.restore(sess, "./inception_v3.ckpt")

f = open(label_dir)
reader = csv.reader(f)
data_files = []
for row in reader:
    data_files.append([row[0], row[2]])



##========================= TRAIN MODELS ================================##

shuffle(data_files)
#train_data = data_files[0:int(len(data_files) * 0.9)]
#val_data = data_files[int(len(data_files) * 0.9):len(data_files)]
# with no val
train_data = data_files

step = 0
for epoch in range(10):
    ##=====Train=====##
    batch_size = 16
    batch_idxs = len(train_data) // batch_size
    shuffle(train_data)
    for idxs in range(batch_idxs):
        batch_files = train_data[idxs * batch_size:(idxs + 1) * batch_size]
        batchx = [load_image(file[0]) for file in batch_files]
        batchy = [file[1] for file in batch_files]
        batch_images = np.array(batchx).astype(np.float32)

        start_time = time.time()
        feed_dict = {x: batch_images, real_y: batchy}
        feed_dict.update(network.all_drop)
        trainacc, trainloss, summary, _ = sess.run([acc, loss,  merged, optim], feed_dict=feed_dict)
        #trainacc, trainloss, _ = sess.run([acc, loss, optim], feed_dict=feed_dict)
        if step % 1 == 0:
            print("Epoch: [%3d/%3d] [%4d/%4d] avgtime %4.4f, train_loss: %.8f, train_acc: %.4f" % (epoch, 10, idxs, batch_idxs, time.time()-start_time, trainloss, trainacc))
            print("Epoch: [%3d/%3d] [%4d/%4d] avgtime %4.4f, train_loss: %.8f, train_acc: %.4f" % (epoch, 10, idxs, batch_idxs, time.time() - start_time, trainloss, trainacc), file=outlogs)
        train_summary_writer.add_summary(summary, step)
        step += 1
    ##===== Val =====##
    # val_batch_size = 16
    # val_batch_idxs = len(val_data) // val_batch_size
    # val_acc = 0
    # start_time = time.time()
    # for idxs in range(val_batch_idxs):
    #     batch_files = val_data[idxs * val_batch_size:(idxs + 1) * val_batch_size]
    #     batchx = [load_image(file[0]) for file in batch_files]
    #     batchy = [file[1] for file in batch_files]
    #     batch_images = np.array(batchx).astype(np.float32)
    #
    #     dp_dict = tl.utils.dict_to_one(network.all_drop)
    #     feed_dict = {x: batch_images, real_y: batchy}
    #     feed_dict.update(dp_dict)
    #     valacc = sess.run([acc], feed_dict=feed_dict)
    #     val_acc += valacc[0]
    #
    # saver.save(sess, logs_dir+"model.ckpt", step)
    #
    # print("Val result: Epoch: [%3d/%3d]  time %4.4f, val_acc: %.4f" % (epoch, 15, time.time() - start_time, val_acc/val_batch_idxs))
    # print("Val result: Epoch: [%3d/%3d]  time %4.4f, val_acc: %.4f" % (epoch, 15, time.time() - start_time, val_acc / val_batch_idxs), file=outlogs)
    # ##=====Save model=====##
    print("[*] Saving checkpoints...")
    tl.files.save_npz(network.all_params, name='./'+model_dir, sess=sess)
    print("[*] Saving checkpoints SUCCESS!")

train_summary_writer.close()
