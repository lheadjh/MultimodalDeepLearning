import tensorflow as tf
import time
import numpy as np
import mdl_data
import sys

GPUNUM = sys.argv[1]
FILEPATH = sys.argv[2]


# Network Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 32
display_step = 1

n_input_img = 4096 # YLI_MED image data input (data shape: 4096, fc7 layer output)
n_hidden_1_img = 1000 # 1st layer num features 1000
n_hidden_2_img = 600 # 2nd layer num features 600

n_input_aud = 100
n_steps_aud = 20  # YLI_MED audio data input (data shape: 2000, mfcc output)
n_hidden_1_aud = 300 # 1st layer num features 1000
#n_hidden_2_aud = 600 # 2nd layer num features 600

n_hidden_1_in = 600
n_hidden_1_out = 256
n_hidden_2_out = 128

n_classes = 10 # YLI_MED total classes (0-9 digits)
dropout = 0.75

with tf.device('/gpu:' + GPUNUM):
    #-------------------------------Struct Graph
    # tf Graph input
    x_aud = tf.placeholder("float", [None, n_steps_aud, n_input_aud])
    x_img = tf.placeholder("float", [None, n_input_img])
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
  
    # Create model
    def Multimodal(_X_aud, _X_img, _w_aud, _b_aud, _w_img, _b_img, _w_out, _b_out, _dropout):
        #------------------------------------aud
        # Permuting batch_size and n_steps
        _X_aud = tf.transpose(_X_aud, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        _X_aud = tf.reshape(_X_aud, [-1, n_input_aud])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        _X_aud = tf.split(0, n_steps_aud, _X_aud)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_1_aud, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_1_aud, forget_bias=1.0)

        # Get lstm cell output
        try:
            aud_outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X_aud, dtype=tf.float32)
        except Exception: # Old TensorFlow version only returns outputs not states
            aud_outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X_aud, dtype=tf.float32)
            
        #------------------------------------Image
        img_layer_1 = tf.nn.relu(tf.add(tf.matmul(_X_img, _w_img['h1']), _b_img['b1'])) #Hidden layer with RELU activation
        drop_1 = tf.nn.dropout(img_layer_1, _dropout)
        img_layer_2 = tf.nn.relu(tf.add(tf.matmul(drop_1, _w_img['h2']), _b_img['b2'])) #Hidden layer with RELU activation
        drop_2 = tf.nn.dropout(img_layer_2, _dropout)
        #img_out = tf.matmul(drop_2, _w_img['out']) + _b_img['out']

        facmat = tf.nn.relu(tf.add(aud_outputs[-1], drop_2))
        
        #out_drop = tf.nn.dropout(merge_sum, _dropout)
        out_layer_1 = tf.nn.relu(tf.add(tf.matmul(facmat, _w_out['h1']), _b_out['b1'])) #Hidden layer with RELU activation
        out_layer_2 = tf.nn.relu(tf.add(tf.matmul(out_layer_1, _w_out['h2']), _b_out['b2'])) #Hidden layer with RELU activation
        
        #return out_drop
        return tf.matmul(out_layer_2, _w_out['out']) + _b_out['out']
    
    # Store layers weight & bias
    w_out = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1_in, n_hidden_1_out])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1_out, n_hidden_2_out])),
        'out': tf.Variable(tf.random_normal([n_hidden_2_out, n_classes]))
    }
    b_out = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1_out])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2_out])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    w_aud = {
        'h1': tf.Variable(tf.random_normal([n_input_aud, 2*n_hidden_1_aud])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1_aud, n_hidden_2_aud])),
        'out': tf.Variable(tf.random_normal([2*n_hidden_1_aud, n_classes]))
    }
    b_aud = {
        'b1': tf.Variable(tf.random_normal([2*n_hidden_1_aud])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2_aud])),
        'out': tf.Variable(tf.random_normal([n_classes]))    
    }
    w_img = {
        'h1': tf.Variable(tf.random_normal([n_input_img, n_hidden_1_img])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1_img, n_hidden_2_img])),
        'out': tf.Variable(tf.random_normal([n_hidden_2_img, n_classes]))    
    }
    b_img = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1_img])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2_img])),
        'out': tf.Variable(tf.random_normal([n_classes]))     
    }

    # Construct model
    pred = Multimodal(x_aud, x_img, w_aud, b_aud, w_img, b_img, w_out, b_out, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    # Initializing the variables
    init = tf.initialize_all_variables()
    '''
    -------------------------------
    Load data
    -------------------------------
    '''
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
    X_aud_train = data.get_aud_X_train()
    y_train = data.get_y_train()
    Y_train = dense_to_one_hot(y_train)

    # Shuffle initial data
    p = np.random.permutation(len(Y_train))
    X_img_train = X_img_train[p]
    X_aud_train = X_aud_train[p]
    Y_train = Y_train[p]

    # Load test data
    X_img_test = data.get_img_X_test()
    X_aud_test = data.get_aud_X_test()
    X_aud_test = X_aud_test.reshape((-1, n_steps_aud, n_input_aud))
    y_test = data.get_y_test()
    Y_test = dense_to_one_hot(y_test)
    '''
    -------------------------------
    Launch the graph
    -------------------------------
    '''
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(Y_train)/batch_size)
            #Loop oveer all batches
            for i in range(total_batch):
                batch_x_aud, batch_x_img, batch_ys, finish = data.next_batch_multi(X_aud_train, X_img_train, Y_train, batch_size, len(Y_train))
                batch_x_aud = batch_x_aud.reshape((batch_size, n_steps_aud, n_input_aud))
                # Fit traning using batch data
                sess.run(optimizer, feed_dict = {x_aud: batch_x_aud, x_img: batch_x_img, y: batch_ys, keep_prob: dropout})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict = {x_aud: batch_x_aud, x_img: batch_x_img, y: batch_ys, keep_prob: 1.}) / total_batch
                #Shuffling
                if finish:
                    p = np.random.permutation(len(Y_train))
                    X_aud_train = X_aud_train[p]
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
        print 'MM_RDN.py'
        print "Accuracy:", accuracy.eval({x_aud: X_aud_test, x_img: X_img_test, y: Y_test, keep_prob: 1.})
