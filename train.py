import time
import theano
import lasagne
import numpy as np
import data_loader
import theano.tensor as T

from lasagne.nonlinearities import leaky_rectify as lrelu, softmax, rectify
from lasagne.layers import batch_norm, DenseLayer, dropout, InputLayer, PadLayer
from lasagne.layers import NonlinearityLayer, ElemwiseSumLayer, ExpressionLayer, GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.regularization import regularize_network_params, l2

def augment_data(X, y):
    """
    Adds flipped by x images to training set
    """
    flipped_X = X[:, :, :, ::-1]

    return np.concatenate([X, flipped_X], axis=0), np.concatenate([y, y])

def convert_data(X, y):
    """
    Does necessary data preprocessing
    """
    converted_X = lasagne.utils.floatX(X)
    converted_y = y.astype('int32')

    return converted_X, converted_y

def residual_block(incoming, increase_dim=False):
    input_num_filters = incoming.output_shape[1]

    if increase_dim:
        num_filters = 2 * input_num_filters
        stride = (2, 2)
    else:
        num_filters = input_num_filters
        stride = (1, 1)

    conv_1 = batch_norm(ConvLayer(incoming, num_filters, (3, 3), stride=stride,
                                  pad='same', nonlinearity=rectify,
                                  W=lasagne.init.HeNormal(gain='relu')))
    conv_2 = batch_norm(ConvLayer(conv_1, num_filters, (3, 3), stride=(1, 1),
                                  pad='same', nonlinearity=None,
                                  W=lasagne.init.HeNormal(gain='relu')))

    if increase_dim:
        identity = ExpressionLayer(incoming, lambda X: X[:, :, ::2, ::2],
                                   lambda s: (s[0], s[1], s[2]//2, s[3]//2))
        incoming = PadLayer(identity, [num_filters // 4, 0, 0], batch_ndim=1)

    layer = NonlinearityLayer(ElemwiseSumLayer([incoming, conv_2]), nonlinearity=rectify)

    return layer

class LasagneSmallResNet(object):

    def __init__(self, size=5, shape=(None, 3, 32, 32), output=10, reg_str=1e-4):
        self.input_var = T.tensor4('X')
        self.target_var = T.ivector('y')
        self._input_shape = shape
        self._output = output
        self._size = size
        self._network = self._build()
        self._loss = self._get_loss()
        self._predict = self._get_predict()
        self._reg_str = reg_str

    def _build(self):
        network = InputLayer(self._input_shape, self.input_var)
        network = batch_norm(ConvLayer(network, 16, (3, 3), stride=(1,1),
                                       nonlinearity=rectify, pad='same',
                                       W=lasagne.init.HeNormal(gain='relu')))
        num_clusters = 4

        for cluster in range(num_clusters):
            for i in range(self._size):

                if cluster != 0 and i == 0:
                    network = residual_block(network, increase_dim=True)
                    continue

                network = residual_block(network)

        network = GlobalPoolLayer(network)
        network = DenseLayer(network, num_units=self._output, W=lasagne.init.HeNormal(),
                             nonlinearity=softmax)

        return network

    def _get_loss(self):
        prediction = lasagne.layers.get_output(self._network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var)

        return loss.mean()

    def _get_predict(self):
        test_prediction = lasagne.layers.get_output(self._network, deterministic=True)

        return theano.function([self.input_var], T.argmax(test_prediction, axis=1))

    def iterate(self, X, y, batch_size, augment=False):

        for idx in range(0, X.shape[0] - batch_size + 1, batch_size):
            mask = slice(idx, idx + batch_size)
            masked_X = X[mask]

            if augment:
                padded = np.pad(masked_X, ((0,0), (0,0), (4,4), (4,4)), mode='constant')
                cropped = np.zeros(masked_X.shape, dtype=np.float32)
                shift_y = np.random.randint(0, high=8, size=(batch_size))
                shift_x = np.random.randint(0, high=8, size=(batch_size))

                for r in xrange(batch_size):
                    cropped[r, :, :, :] = padded[r, :, shift_y[r]:(shift_y[r] + 32), shift_x[r]:(shift_x[r] + 32)]

                masked_X = cropped

            yield masked_X, y[mask]

    def train(self, X, y, lr=0.1, lrd=0.99, n_epoch=50, batch_size=256,
              verbose=True, X_test=None, y_test=None):
        """
        Train network

        Inputs:
        - X: Array if data, of shape(N, 3, 32, 32)
        - y: Array of labels, of shape (N,)
        - lr: Learning rate
        - lrd: Learning rate decay
        - n_epoch: The number of epochs
        - batch_size: data split size to avoid using too much memory
        - verbose: Boolean; if set to false then no output will be printed
        """
        #param update
        training_loss = self._loss + self._reg_str * regularize_network_params(self._network, l2)
        shared_lr = theano.shared(lasagne.utils.floatX(lr))
        params = lasagne.layers.get_all_params(self._network, trainable=True)
        updates = lasagne.updates.momentum(training_loss, params, learning_rate=shared_lr, momentum=0.9)

        #compile
        train_fn = theano.function([self.input_var, self.target_var], training_loss,
                                   updates=updates)

        #loss expression for validation
        test_prediction = lasagne.layers.get_output(self._network, deterministic=True)
        test_accuracy = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),
                          dtype=theano.config.floatX)

        val_fn = theano.function([self.input_var, self.target_var], [self._loss, test_accuracy])

        #train
        if verbose:
            print("Start training")

        num_train = X.shape[0]
        iterations_per_epoch = num_train / batch_size

        if verbose:
            print("model params: {}".format(lasagne.layers.count_params(self._network, trainable=True)))
            print(iterations_per_epoch)

        for epoch in range(n_epoch):
            loss, test_loss, test_acc, test_batches = 0, 0, 0, 0
            start_time = time.time()

            train_indices = np.arange(num_train)
            np.random.shuffle(train_indices)
            X_train = X[train_indices]
            y_train = y[train_indices]

            for X_batch, y_batch in self.iterate(X_train, y_train, batch_size, augment=True):
                loss += train_fn(X_batch, y_batch)

            for X_test_batch, y_test_batch in self.iterate(X_test, y_test, batch_size=500):
                tloss, tacc = val_fn(X_test_batch, y_test_batch)
                test_batches += 1
                test_loss += tloss
                test_acc += tacc

            lrate = shared_lr.get_value()
            lrate *= lrd

            if epoch == 40 or epoch == 60:
                lrate *= 0.1

            shared_lr.set_value(lasagne.utils.floatX(lrate))

            if verbose:
                print("Epoch {} took {:.3f}s (learning rate {})".format(epoch + 1, time.time() - start_time, lrate))
                print("  training loss: {:.6f}".format(loss / iterations_per_epoch))
                print("  test loss:     {:.6f}".format(test_loss / test_batches))
                print("  test accuracy: {:.4f}".format(test_acc / test_batches))

    def predict(self, X):
        """
        Predicts classes of input X

        Inputs:
        - X: Array of shape(N, 3, 32, 32)

        Returns:
        - predicted: Array of labels, of shape (N,)
        """
        return self._predict(lasagne.utils.floatX(X))

    def save(self, filename):
        np.savez(filename, *lasagne.layers.get_all_param_values(self._network))

    def load(self, filename):

        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(network, param_values)

if __name__ == '__main__':
    data = data_loader.get_CIFAR10_data()

    for k, v in data.iteritems():
        print '%s: ' % k, v.shape

    X = data['X_train']
    y = data['y_train']
    X, y = convert_data(*augment_data(X, y))
    X_test, y_test = convert_data(data['X_test'], data['y_test'])
    X_val, y_val = convert_data(*augment_data(data['X_val'], data['y_val']))
    print('prepared X, y:', X.shape, y.shape)
    smallResNet = LasagneSmallResNet()
    smallResNet.train(X, y, n_epoch=100, lrd=.999, X_test=X_test, y_test=y_test)
    predicted = smallResNet.predict(X_val)
    accuracy = (predicted == y_val).mean()
    smallResNet.save("model{:.3f}".format(accuracy))

    print("validation accuracy:", accuracy)
