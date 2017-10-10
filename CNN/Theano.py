import os
import sys
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import gzip
import numpy
from LeNetConvPoolLayer import LeNetConvPoolLayer
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
import time

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'data.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'data.pkl.gz':
        print "NO This File"
    

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            x,y = pickle.load(f, encoding='latin1')
        except:
            x,y = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = shared_y.flatten()
        
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    train_set_x, train_set_y=shared_dataset([x[0::2],y[0::2]])
    valid_set_x, valid_set_y=shared_dataset([x[0::5],y[0::5]])
    test_set_x, test_set_y=shared_dataset([x[0::3],y[0::3]])    
    #test_set_x, test_set_y = shared_dataset(test_set)  
    #valid_set_x, valid_set_y = shared_dataset(valid_set)  
    #train_set_x, train_set_y = shared_dataset(train_set)  
  
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),  
            (test_set_x, test_set_y)]  
    return rval  



def CNN(learning_rate, n_epochs,nkerns, batch_size):
    rng = numpy.random.RandomState(23455)
    datasets=load_data('data.pkl.gz')
    
    train_set_x, train_set_y=datasets[0]
    valid_set_x, valid_set_y=datasets[1]
    test_set_x, test_set_y=datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] 
    n_train_batches /= batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] 
    n_valid_batches /= batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] 
    n_test_batches /= batch_size
    
    print('... building the model')
    print test_set_y[0]
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    
    # generate symbolic variables for input (x and y represent a
        # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    layer0_input = x.reshape((batch_size, 1, 50, 50))
    
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 50, 50),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2) 
        
    )
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 23, 23),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 10 * 10,
        n_out=batch_size,
        activation=T.tanh
    )
    
    layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=4)
    cost = layer3.negative_log_likelihood(y)
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(  
        [index],  
        layer3.errors(y),  
        givens={  
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],  
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]  
        }  
    )      
    params = layer3.params + layer2.params + layer1.params + layer0.params
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print '... training'  
    patience = 10000    
    patience_increase = 2    
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2) 
    best_validation_loss = numpy.inf 
    best_iter = 0
    test_score = 0.  
    start_time = time.clock()    
    epoch = 0  
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):  
        epoch = epoch + 1  
        for minibatch_index in xrange(n_train_batches):  
    
            iter = (epoch - 1) * n_train_batches + minibatch_index  
    
            if iter % 100 == 0:  
                print 'training @ iter = ', iter  
            cost_ij = train_model(minibatch_index)    
    #cost_ij 没什么用，后面都没有用到,只是为了调用train_model，而train_model有返回值  
            if (iter + 1) % validation_frequency == 0:  
    
                # compute zero-one loss on validation set  
                validation_losses = [validate_model(i) for i  
                                     in xrange(n_valid_batches)]  
                this_validation_loss = numpy.mean(validation_losses)  
                print('epoch %i, minibatch %i/%i, validation error %f %%' %  
                      (epoch, minibatch_index + 1, n_train_batches,  
                       this_validation_loss * 100.))  
    
    
                if this_validation_loss < best_validation_loss:  
    
    
                    if this_validation_loss < best_validation_loss * improvement_threshold:  
                        patience = max(patience, iter * patience_increase)  
    
    
                    best_validation_loss = this_validation_loss  
                    best_iter = iter  
    
    
                    test_losses = [  
                        test_model(i)  
                        for i in xrange(n_test_batches)  
                    ]  
                    test_score = numpy.mean(test_losses)  
                    print(('     epoch %i, minibatch %i/%i, test error of '  
                           'best model %f %%') %  
                          (epoch, minibatch_index + 1, n_train_batches,  
                           test_score * 100.))  
    
            if patience <= iter:  
                done_looping = True  
                break  

    end_time = time.clock()  
    print('Optimization complete.')  
    print('Best validation score of %f %% obtained at iteration %i, '  
          'with test performance %f %%' %  
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))  
    print >> sys.stderr, ('The code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))            
                        
                    
CNN(learning_rate=0.1, n_epochs=50, nkerns=[15, 15], batch_size=10)