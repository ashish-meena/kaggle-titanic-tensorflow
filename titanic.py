import tensorflow as tf
import pandas as pd
from titanic_dataset import TitanicDataSet

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
LEARNING_RATE = 0.001
EPOCH_NUM = 200000
BATCH_SIZE = 100

titanicDataSet = TitanicDataSet(TRAIN_FILE, TEST_FILE)

train_data = titanicDataSet.train
test_data = titanicDataSet.test

x = tf.placeholder(tf.float32, [None, 7], name='Input')
y = tf.placeholder(tf.float32, [None, 2], name='OutPut')

W = tf.Variable(tf.zeros([7, 2]), name='Weights')
b = tf.Variable(tf.zeros([2]), name='Bias')
Ylogits = tf.matmul(x, W) + b
pred = tf.nn.softmax(Ylogits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    training_dataset_size = titanicDataSet.train.shape[0]
    for epoch in range(EPOCH_NUM):
        total_cost = 0
        total_batch = int(training_dataset_size / BATCH_SIZE)

        for i in range(total_batch):
            batch_xs, batch_ys = titanicDataSet.next_batch(BATCH_SIZE)
            _,c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            total_cost += c

        print ("epoch ", epoch, " completed", " cost=", total_cost)
    indexes = titanicDataSet.test.index.values
    feed_dict = {x: titanicDataSet.test}
    predict_proba = pred.eval(feed_dict)
    predictions = tf.argmax(predict_proba, dimension=1).eval()

    with open("kaggle.csv", "w") as f:
        f.write("PassengerId,Survived\n")
        for index, prediction in zip(indexes, predictions):
            f.write("{0},{1}\n".format(index, prediction))
