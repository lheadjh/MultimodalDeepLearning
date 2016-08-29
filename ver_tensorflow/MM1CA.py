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
    keep_tr = tf.placeholder(tf.float32)

    def calculatCA(_tp1, _tp2, size, _b_size):
        first = True
        tp1 = tf.split(0, _b_size, _tp1)
        tp2 = tf.split(0, _b_size, _tp2)
        for i in range(_b_size):
            input1 = tf.reshape(tp1[i], shape=[size, 1])
            input2 = tf.reshape(tp2[i], shape=[size, 1])

            upper = tf.matmul(tf.transpose(tf.sub(input1, tf.reduce_mean(input1))), tf.sub(input2, tf.reduce_mean(input2)))        
            _tp1 = tf.reduce_sum(tf.mul(tf.sub(input1, tf.reduce_mean(input1)), tf.sub(input1, tf.reduce_mean(input1))))
            _tp2 = tf.reduce_sum(tf.mul(tf.sub(input2, tf.reduce_mean(input2)), tf.sub(input2, tf.reduce_mean(input2))))
            down = tf.sqrt(tf.mul(_tp1, _tp2))
            factor = tf.abs(tf.div(upper, down))
            
            if first:
                output = factor
                first = False
            else:
                output = tf.concat(1, [output, factor])

        return tf.transpose(output)
    
    # Create model
    def multilayer_perceptron(_X_aud, _X_img, _w_aud, _b_aud, _w_img, _b_img, _w_out, _b_out, _dropout, _b_size):
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

        '''
        Merge with CA
        '''
        factor = calculatCA(aud_layer_2, drop_2, 600, _b_size)
        factor = tf.reshape(tf.diag(factor), shape=[_b_size, _b_size])
        merge_sum = tf.add(aud_layer_2, drop_2)
        facmat = tf.nn.relu(tf.matmul(factor, merge_sum))
   
    
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
    pred = multilayer_perceptron(x_aud, x_img, w_aud, b_aud, w_img, b_img, w_out, b_out, keep_prob, batch_size)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    # Initializing the variables
    init = tf.initialize_all_variables()

    #-------------------------------Load data
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
    y_test = data.get_y_test()
    Y_test = dense_to_one_hot(y_test)

    #-------------------------------Launch the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(Y_train)/batch_size)
            #Loop oveer all batches
            for i in range(total_batch):
                batch_x_aud, batch_x_img, batch_ys, finish = data.next_batch_multi(X_aud_train, X_img_train, Y_train, batch_size, len(Y_train))
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
        batch_size = 1
        pred = multilayer_perceptron(x_aud, x_img, w_aud, b_aud, w_img, b_img, w_out, b_out, keep_prob, batch_size)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        test = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        total = 0
        correct = 0
        for i in range(int(len(Y_test)/batch_size)):
            total += batch_size
            batch_x_aud, batch_x_img, batch_ys, finish = data.next_batch_multi(X_aud_test, X_img_test, Y_test, batch_size, len(Y_test))
            correct += test.eval({x_aud: batch_x_aud, x_img: batch_x_img, y: batch_ys, keep_prob: 1.})
        print int(len(Y_test)/batch_size)
        print correct
        print total
        print float(correct/total)
        print 'MM1CA.py'
        # Calculate accuracy
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))     
        #for epoch in range(Y_test):
        #    print "Accuracy:", accuracy.eval({x_aud: X_aud_test, x_img: X_img_test, y: Y_test, keep_prob: 1.})
