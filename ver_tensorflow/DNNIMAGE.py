import tensorflow as tf
import time
import numpy as np
import mdl_data
import sys

GPUNUM = sys.argv[1]
FILEPATH = sys.argv[2]

with tf.device('/gpu:' + GPUNUM):
    #Source reference: https://github.com/aymericdamien/TensorFlow-Examples.git/input_data.py
    def dense_to_one_hot(labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    # Load data
    data = mdl_data.YLIMED('YLIMED_info.csv', FILEPATH + '/YLIMED150924/audio/mfcc20', FILEPATH + '/YLIMED150924/keyframe/fc7')
    X_img_train = data.get_img_X_train()
    y_train = data.get_y_train()
    Y_train = dense_to_one_hot(y_train)

    # Shuffle initial data
    p = np.random.permutation(len(Y_train))
    X_img_train = X_img_train[p]
    Y_train = Y_train[p]

    # Load test data
    X_img_test = data.get_img_X_test()
    y_test = data.get_y_test()
    Y_test = dense_to_one_hot(y_test)

    learning_rate = 0.001
    training_epochs = 100
    batch_size = 256
    display_step = 1

    # Network Parameters
    n_hidden_1 = 1000 # 1st layer num features
    n_hidden_2 = 600 # 2nd layer num features
    n_input_img = 4096 # YLI_MED image data input (data shape: 4096, fc7 layer output)
    n_classes = 10 # YLI_MED total classes (0-9 digits)
    dropout = 0.75

    #image part

    x = tf.placeholder("float", [None, n_input_img])
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # Create model
    def multilayer_perceptron(_X, _weights, _biases, _dropout):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
        drop_1 = tf.nn.dropout(layer_1, _dropout)
        layer_2 = tf.nn.relu(tf.add(tf.matmul(drop_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
        drop_2 = tf.nn.dropout(layer_2, _dropout)
        return tf.matmul(drop_2, _weights['out']) + _biases['out']

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input_img, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graphe
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(Y_train)/batch_size)
            #Loop oveer all batches
            for i in range(total_batch):
                batch_xs, batch_ys, finish = data.next_batch(X_img_train, Y_train, batch_size, len(Y_train))
                # Fit traning using batch data
                sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: dropout})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 1.}) / total_batch
                #Shuffling
                if finish:
                    p = np.random.permutation(len(Y_train))
                    X_img_train = X_img_train[p]
                    Y_train = Y_train[p]
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        print "Optimization Finished!"

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print "Accuracy:", accuracy.eval({x: X_img_test, y: Y_test, keep_prob: 1.})
        print 'DNNIMAGE.py'