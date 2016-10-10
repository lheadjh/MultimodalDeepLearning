import tensorflow as tf
import time
import numpy as np
import mdl_data
import sys

GPUNUM = sys.argv[1]
FILEPATH = sys.argv[2]


# Network Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 256
display_step = 1

n_input_img = 4096 # YLI_MED image data input (data shape: 4096, fc7 layer output)
n_hidden_1_img = 1000 # 1st layer num features 1000
n_hidden_2_img = 600 # 2nd layer num features 600

n_input_aud = 2000 # YLI_MED audio data input (data shape: 2000, mfcc output)
n_hidden_1_aud = 1000 # 1st layer num features 1000
n_hidden_2_aud = 600 # 2nd layer num features 600

n_hidden_1_in = 600
n_hidden_1_out = 256
n_hidden_2_out = 128

n_classes = 10 # YLI_MED total classes (0-9 digits)
dropout = 0.75

with tf.device('/gpu:' + GPUNUM):
    #-------------------------------Struct Graph
    # tf Graph input
    x_aud = tf.placeholder("float", [None, n_input_aud])
    x_img = tf.placeholder("float", [None, n_input_img])
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
  
    # Create model
    def multimodal(_X_aud, _X_img, _w_aud, _b_aud, _w_img, _b_img, _w_out, _b_out, _dropout):
        #aud
        aud_layer_1 = tf.nn.relu(tf.add(tf.matmul(_X_aud, _w_aud['h1']), _b_aud['b1'])) #Hidden layer with RELU activation
        aud_layer_2 = tf.nn.relu(tf.add(tf.matmul(aud_layer_1, _w_aud['h2']), _b_aud['b2'])) #Hidden layer with RELU activation
        #aud_out = tf.matmul(aud_layer_2, _w_aud['out']) + _b_aud['out']
        #Image
        img_layer_1 = tf.nn.relu(tf.add(tf.matmul(_X_img, _w_img['h1']), _b_img['b1'])) #Hidden layer with RELU activation
        drop_1 = tf.nn.dropout(img_layer_1, _dropout)
        img_layer_2 = tf.nn.relu(tf.add(tf.matmul(drop_1, _w_img['h2']), _b_img['b2'])) #Hidden layer with RELU activation
        drop_2 = tf.nn.dropout(img_layer_2, _dropout)
        #img_out = tf.matmul(drop_2, _w_img['out']) + _b_img['out']

        tmp_pool =tf.add(aud_layer_2, drop_2)
        tmp_pool = tf.reduce_sum(tmp_pool, 0, keep_dims=True)
        
        out_layer_1 = tf.nn.relu(tf.add(tf.matmul(tmp_pool, _w_out['h1']), _b_out['b1'])) #Hidden layer with RELU activation
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
        'h1': tf.Variable(tf.random_normal([n_input_aud, n_hidden_1_aud])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1_aud, n_hidden_2_aud])),
        'out': tf.Variable(tf.random_normal([n_hidden_2_aud, n_classes]))
    }
    b_aud = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1_aud])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2_aud])),
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
    pred = multimodal(x_aud, x_img, w_aud, b_aud, w_img, b_img, w_out, b_out, keep_prob)

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
    Vid_training = data.get_vid_info('Training')
    Vid_test = data.get_vid_info('Test')

    # Shuffle initial data
    p = np.random.permutation(len(Vid_training))
    Vid_training = Vid_training[p]
    p = np.random.permutation(len(Vid_test))
    Vid_test = Vid_test[p]

    # Load test data
    #X_img_test = data.get_img_X_test()
    #X_aud_test = data.get_aud_X_test()
    #y_test = data.get_y_test()
    #Y_test = dense_to_one_hot(y_test)
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
            total_batch = len(Vid_training)
            #Loop oveer all batches
            for i in range(total_batch):
                batch_x_aud = data.get_vid_data(Vid_training[i], 'Aud')
                batch_x_img = data.get_vid_data(Vid_training[i], 'Img')
                batch_y = np.asarray([int(Vid_training[i].split()[2])])
                batch_y = dense_to_one_hot(batch_y)
                # Fit traning using batch data
                sess.run(optimizer, feed_dict = {x_aud: batch_x_aud, x_img: batch_x_img, y: batch_y, keep_prob: dropout})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict = {x_aud: batch_x_aud, x_img: batch_x_img, y: batch_y, keep_prob: 1.})
            #Shuffling
            p = np.random.permutation(len(Vid_training))
            Vid_training = Vid_training[p]
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        print "Optimization Finished!"

        # Test model
        correct = 0
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        total_batch = len(Vid_test)
        for i in range(total_batch):
            batch_x_aud = data.get_vid_data(Vid_test[i], 'Aud')
            batch_x_img = data.get_vid_data(Vid_test[i], 'Img')
            batch_y = np.asarray([int(Vid_test[i].split()[2])])
            batch_y = dense_to_one_hot(batch_y)
            if correct_prediction.eval({x_aud: batch_x_aud, x_img: batch_x_img, y: batch_y, keep_prob: 1.}):
                correct+=1
            if i % 100 == 0:
                print i
        print "Accuracy:", correct, len(Vid_test)
        print "MM_video.py"