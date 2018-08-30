import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

#权重初始化
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)
#需要将数据转化为[image_num, x, y, depth]格式
def extract_images_and_labels(dataset, validation = False):
    images = dataset[:, 1:].reshape(-1, 28, 28, 1)
    labels_dense = dataset[:, 0]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * 10
    labels_one_hot = np.zeros((num_labels, 10))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    if validation:
        num_images = images.shape[0]
        divider = num_images - 200
        return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]
    else:
        return images, labels_one_hot

def extract_images(dataset):
    return dataset.reshape(-1, 28*28)

train_data_file = 'train.csv'
test_data_file = 'test.csv'

train_data = pd.read_csv(train_data_file).as_matrix().astype(np.uint8)
test_data = pd.read_csv(test_data_file).as_matrix().astype(np.uint8)

train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
test_images = extract_images(test_data)

#创建DataSet对象
train = DataSet(train_images,
                train_labels,
                dtype = np.float32,
                reshape = True)

validation = DataSet(val_images,
                val_labels,
                dtype = np.float32,
                reshape = True)

test = test_images

def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

#第一层卷积
W_conv1 = weight_variable([4, 4, 1, 20])
b_conv1 = bias_variable([20])

#为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
x_image = tf.reshape(x,[-1,28,28,1])

#我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5, 5, 20, 800])
b_conv2 = bias_variable([800])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_3x3(h_conv2)

#密集连接层
W_fc1 = weight_variable([5 * 5 * 800, 150])
b_fc1 = bias_variable([150])

h_pool3_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 800])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练和评估模型
MAX_ITERATION = 12000;
BATCH_SIZE = 50;

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print('Starting Training Process...')
maxAccuracy = 0.0
for i in range(MAX_ITERATION):
  batch = train.next_batch(BATCH_SIZE)

  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    if train_accuracy > maxAccuracy:
        maxAccuracy = train_accuracy
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("Finishing Training Process...")

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: validation.images, y_: validation.labels, keep_prob: 1.0}))


print('Writing Output To csv File...')
predicted_labels = np.zeros(test_images.shape[0])
test_images = np.multiply(test_images, 1.0 / 255.0)
prediction = tf.argmax(y_conv, 1)
for i in range(0, test_images.shape[0]//BATCH_SIZE):
    predicted_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = prediction.eval(feed_dict={x: test_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], keep_prob: 1.0})

# Writes output
np.savetxt('Submission_2_12000.csv', np.c_[range(1, len(test_images)+1), predicted_labels],
           delimiter=',', header='ImageId,Label', comments='', fmt='%d')
sess.close()
