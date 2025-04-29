#!/usr/bin/env python
# coding: utf-8

# # A4 Classification of Hand-Drawn Digits
# 
# In this assignment, you will define a new class named `NeuralNetworkClassifier` that extends the `NeuralNetwork` class provided here and is the solution to Assignment A2.  You will use `NeuralNetworkClassifier` to train a classifier of hand-drawn digits.
# 
# You will also define the function `confusion_matrix`.

# ## `NeuralNetwork` class

# In[11]:


import matplotlib.pyplot as plt


# The following code cell will write its contents to `optimizers.py` so the `import optimizers` statement in the code cell after it will work correctly.

# In[12]:


get_ipython().run_cell_magic('writefile', 'optimizers.py', "import numpy as np\n\n######################################################################\n## class Optimizers()\n######################################################################\n\nclass Optimizers():\n\n    def __init__(self, all_weights):\n        '''all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector'''\n        \n        self.all_weights = all_weights\n\n        # The following initializations are only used by adam.\n        # Only initializing m, v, beta1t and beta2t here allows multiple calls to adam to handle training\n        # with multiple subsets (batches) of training data.\n        self.mt = np.zeros_like(all_weights)\n        self.vt = np.zeros_like(all_weights)\n        self.beta1 = 0.9\n        self.beta2 = 0.999\n        self.beta1t = 1\n        self.beta2t = 1\n\n        \n    def sgd(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, verbose=True, error_convert_f=None):\n        '''\nerror_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.\ngradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error\n            with respect to each weight.\nerror_convert_f: function that converts the standardized error from error_f to original T units.\n        '''\n\n        error_trace = []\n        epochs_per_print = n_epochs // 10\n\n        for epoch in range(n_epochs):\n\n            error = error_f(*fargs)\n            grad = gradient_f(*fargs)\n\n            # Update all weights using -= to modify their values in-place.\n            self.all_weights -= learning_rate * grad\n\n            if error_convert_f:\n                error = error_convert_f(error)\n            error_trace.append(error)\n\n            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):\n                print(f'sgd: Epoch {epoch+1:d} Error={error:.5f}')\n\n        return error_trace\n\n    def adam(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, verbose=True, error_convert_f=None):\n        '''\nerror_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.\ngradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error\n            with respect to each weight.\nerror_convert_f: function that converts the standardized error from error_f to original T units.\n        '''\n\n        alpha = learning_rate  # learning rate called alpha in original paper on adam\n        epsilon = 1e-8\n        error_trace = []\n        epochs_per_print = n_epochs // 10\n\n        for epoch in range(n_epochs):\n\n            error = error_f(*fargs)\n            grad = gradient_f(*fargs)\n\n            self.mt[:] = self.beta1 * self.mt + (1 - self.beta1) * grad\n            self.vt[:] = self.beta2 * self.vt + (1 - self.beta2) * grad * grad\n            self.beta1t *= self.beta1\n            self.beta2t *= self.beta2\n\n            m_hat = self.mt / (1 - self.beta1t)\n            v_hat = self.vt / (1 - self.beta2t)\n\n            # Update all weights using -= to modify their values in-place.\n            self.all_weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)\n    \n            if error_convert_f:\n                error = error_convert_f(error)\n            error_trace.append(error)\n\n            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):\n                print(f'Adam: Epoch {epoch+1:d} Error={error:.5f}')\n\n        return error_trace\n\nif __name__ == '__main__':\n\n    import matplotlib.pyplot as plt\n    plt.ion()\n\n    def parabola(wmin):\n        return ((w - wmin) ** 2)[0]\n\n    def parabola_gradient(wmin):\n        return 2 * (w - wmin)\n\n    w = np.array([0.0])\n    optimizer = Optimizers(w)\n\n    wmin = 5\n    optimizer.sgd(parabola, parabola_gradient, [wmin],\n                  n_epochs=500, learning_rate=0.1)\n\n    print(f'sgd: Minimum of parabola is at {wmin}. Value found is {w}')\n\n    w = np.array([0.0])\n    optimizer = Optimizers(w)\n    optimizer.adam(parabola, parabola_gradient, [wmin],\n                   n_epochs=500, learning_rate=0.1)\n    \n    print(f'adam: Minimum of parabola is at {wmin}. Value found is {w}')\n")


# In[13]:


import numpy as np
import optimizers
import sys  # for sys.float_info.epsilon

######################################################################
## class NeuralNetwork()
######################################################################

class NeuralNetwork():


    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs, activation_function='tanh'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation_function = activation_function

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_per_layer == 0 or n_hiddens_per_layer == [] or n_hiddens_per_layer == [0]:
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        # Initialize weights, by first building list of all weight matrix shapes.
        n_in = n_inputs
        shapes = []
        for nh in self.n_hiddens_per_layer:
            shapes.append((n_in + 1, nh))
            n_in = nh
        shapes.append((n_in + 1, n_outputs))

        # self.all_weights:  vector of all weights
        # self.Ws: list of weight matrices by layer
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)

        # Define arrays to hold gradient values.
        # One array for each W array with same shape.
        self.all_gradients, self.dE_dWs = self.make_weights_and_views(shapes)

        self.trained = False
        self.total_epochs = 0
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None


    def make_weights_and_views(self, shapes):
        # vector of all weights built by horizontally stacking flatenned matrices
        # for each layer initialized with uniformly-distributed values.
        all_weights = np.hstack([np.random.uniform(size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements from vector of all weights
        # into correct shape for each layer.
        views = []
        start = 0
        for shape in shapes:
            size =shape[0] * shape[1]
            views.append(all_weights[start:start + size].reshape(shape))
            start += size
        return all_weights, views


    # Return string that shows how the constructor was called
    def __repr__(self):
        return f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_per_layer}, {self.n_outputs}, \'{self.activation_function}\')'


    # Return string that is more informative to the user about the state of this neural network.
    def __str__(self):
        result = self.__repr__()
        if len(self.error_trace) > 0:
            return self.__repr__() + f' trained for {len(self.error_trace)} epochs, final training error {self.error_trace[-1]:.4f}'


    def train(self, X, T, n_epochs, learning_rate, method='sgd', verbose=True):
        '''
train: 
  X: n_samples x n_inputs matrix of input samples, one per row
  T: n_samples x n_outputs matrix of target output values, one sample per row
  n_epochs: number of passes to take through all samples updating weights each pass
  learning_rate: factor controlling the step size of each update
  method: is either 'sgd' or 'adam'
        '''

        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            
        # Standardize X and T
        X = (X - self.Xmeans) / self.Xstds
        T = (T - self.Tmeans) / self.Tstds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = optimizers.Optimizers(self.all_weights)

        # Define function to convert value from error_f into error in original T units, 
        # but only if the network has a single output. Multiplying by self.Tstds for 
        # multiple outputs does not correctly unstandardize the error.
        if len(self.Tstds) == 1:
            error_convert_f = lambda err: (np.sqrt(err) * self.Tstds)[0] # to scalar
        else:
            error_convert_f = lambda err: np.sqrt(err)[0] # to scalar
            

        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        verbose=True,
                                        error_convert_f=error_convert_f)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         verbose=True,
                                         error_convert_f=error_convert_f)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        # Return neural network object to allow applying other methods after training.
        #  Example:    Y = nnet.train(X, T, 100, 0.01).use(X)
        return self

    def relu(self, s):
        s[s < 0] = 0
        return s

    def grad_relu(self, s):
        return (s > 0).astype(int)
    
    def forward_pass(self, X):
        '''X assumed already standardized. Output returned as standardized.'''
        self.Ys = [X]
        for W in self.Ws[:-1]:
            if self.activation_function == 'relu':
                self.Ys.append(self.relu(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
            else:
                self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        Ys = self.forward_pass(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        '''Assumes forward_pass just called with layer outputs in self.Ys.'''
        error = T - self.Ys[-1]
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if self.activation_function == 'relu':
                delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
            else:
                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
        return self.all_gradients

    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = (X - self.Xmeans) / self.Xstds
        Ys = self.forward_pass(X)
        Y = Ys[-1]
        # Unstandardize output Y before returning it
        return Y * self.Tstds + self.Tmeans


# In[14]:


X = np.arange(100).reshape((-1, 1))
T = (X - 20) ** 3 / 300000

hiddens = [10]
nnet = NeuralNetwork(X.shape[1], hiddens, T.shape[1])
nnet.train(X, T, 250, 0.01, method='adam')

plt.subplot(1, 2, 1)
plt.plot(nnet.error_trace)

plt.subplot(1, 2, 2)
plt.plot(T, label='T')
plt.plot(nnet.use(X), label='Y')
plt.legend()


# ## Your `NeuralNetworkClassifier` class

# Complete the following definition of `NeuralNetworkClassifier` as discussed in class. You will need to override the functions
# 
# * `train`
# * `error_f`
# * `gradient_f`
# * `use`
# 
# and define the following new functions
# 
# * `makeIndicatorVars`
# * `softmax`

# In[20]:


class NeuralNetworkClassifier(NeuralNetwork):

    def train(self, X, T, n_epochs, learning_rate, method='sgd', verbose=True):
        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing 
        
        # Standardize X and T
        X = (X - self.Xmeans) / self.Xstds
        TtrainI = self.makeIndicatorVars(T)
        print("X shape: ", X.shape)
        print("T shape: ", T.shape)

        self.n_classes = TtrainI.shape[1]
        self.all_classes = np.unique(T)

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = optimizers.Optimizers(self.all_weights)
        to_likelihood = lambda nll: np.exp(-nll)
            
        if method == 'sgd':

            error_trace = optimizer.sgd(self.neg_log_likelihood, self.gradient_neg_log_likelihood,
                                        fargs=[X, TtrainI], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        verbose=True,
                                        error_convert_f=to_likelihood)

        elif method == 'adam':

            error_trace = optimizer.adam(self.neg_log_likelihood, self.gradient_neg_log_likelihood,
                                         fargs=[X, TtrainI], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         verbose=True,
                                         error_convert_f=to_likelihood)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        # Return neural network object to allow applying other methods after training.
        #  Example:    Y = nnet.train(X, T, 100, 0.01).use(X)
        return self
    
    def makeIndicatorVars(self, T):
        # Make sure T is two-dimensional. Should be nSamples x 1.
        if T.ndim == 1:
            T = T.reshape((-1, 1))
        return (T == np.unique(T)).astype(int)
    
    def softmax(self, X):
        Ys = self.forward_pass(X)
        fs = np.exp(Ys[-1])  # N x K
        denom = np.sum(fs, axis=1).reshape((-1, 1))
        gs = fs / denom
        return gs
    
    def neg_log_likelihood(self, X, T):
        Y = self.softmax(X)
        return -np.mean(T * np.log(Y))
    
    def gradient_neg_log_likelihood(self, X, T):
        '''Assumes forward_pass just called with layer outputs in self.Ys.'''
        error = T - self.softmax(X)
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if self.activation_function == 'relu':
                delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
            else:
                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
        return self.all_gradients
    
    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = (X - self.Xmeans) / self.Xstds
        Y = self.softmax(X) 
        classes = self.all_classes[np.argmax(Y, axis=1)].reshape(-1, 1)
        return classes,Y


# Here is a simple test of your new class.  For inputs from 0 to 100, classify values less than or equal to 25 as Class Label 25, greater than 25 and less than or equal to 75 as Class Label 75, and greater than 75 as Class Label 100. 

# In[16]:


X = np.arange(100).reshape((-1, 1))
T = X.copy()
T[T <= 25] = 25
T[np.logical_and(25 < T, T <= 75)] = 75
T[T > 75] = 100

plt.plot(X, T, 'o-')
plt.xlabel('X')
plt.ylabel('Class');


# In[21]:


hiddens = [10]
nnet = NeuralNetworkClassifier(X.shape[1], hiddens, len(np.unique(T)))
nnet.train(X, T, 200, 0.01, method='adam', verbose=True)

Y_classes, Y_probs = nnet.use(X)

percent_correct = 100 * np.mean(Y_classes == T)
print(percent_correct)

plt.subplot(1, 2, 1)
plt.plot(nnet.error_trace)
plt.xlabel('Epoch')
plt.ylabel('Likelihood')

plt.subplot(1, 2, 2)
plt.plot(T + 5, 'o-', label='T + 5')  # to see, when predicted overlap T very closely
plt.plot(nnet.use(X)[0], 'o-', label='Y')
plt.legend()


# ## Now for the Hand-Drawn Digits
# 
# We will use a bunch (50,000) images of hand drawn digits from [this deeplearning.net site](http://deeplearning.net/tutorial/gettingstarted.html).  Download `mnist.pkl.gz`. 
# 
# deeplearning.net goes down a lot.  If you can't download it from there you can try getting it from [here](https://gitlab.cs.washington.edu/colinxs/neural_nets/blob/master/mnist.pkl.gz).
# 
# This pickle file includes data already partitioned into training, validation, and test sets.  To read it into python, use the following steps

# In[8]:


import pickle
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Xtrain = train_set[0]
Ttrain = train_set[1].reshape(-1, 1)

Xval = valid_set[0]
Tval = valid_set[1].reshape(-1, 1)

Xtest = test_set[0]
Ttest = test_set[1].reshape(-1, 1)

print(Xtrain.shape, Ttrain.shape,  Xval.shape, Tval.shape,  Xtest.shape, Ttest.shape)


# In[ ]:


Ttrain[:10]


# Those must be the digits.  What the heck is in those 784 columns in the input matrices?

# In[ ]:


plt.plot(Xtrain[0, :]);


# Well, values between 0 and 1.  That doesn't help much.  These are actually intensity values for 784 pixels in an image.
# 
# How can we rearrange these values into an image to be displayed?  We must first figure out how many columns and rows the image would have.  Perhaps the image is a square image, with equal numbers of rows and columns.

# In[ ]:


import math
math.sqrt(784)


# Ah, cool.

# In[ ]:


28 * 28


# Ok Let's reshape it and look at the numbers.

# In[ ]:


image0 = Xtrain[0, :]
image0 = image0.reshape(28, 28)
image0


# Not that helpful.  Ok, let's use `matplotlib` to make an image display.

# In[ ]:


plt.imshow(image0);


# Humm.  Try a grayscale color map.

# In[ ]:


plt.imshow(image0, cmap='gray');


# With a little more work, we can make it look like a pencil drawing.

# In[ ]:


plt.imshow(-image0, cmap='gray')  # notice the negative sign
plt.axis('off');


# Looks like a 5.  What class label is associated with this image?

# In[ ]:


Ttrain[0]


# Okay.  Makes sense.  Let's look at the first 100 images and their labels, as plot titles.

# In[ ]:


plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(-Xtrain[i, :].reshape(28, 28), cmap='gray')
    plt.title(Ttrain[i, 0])
    plt.axis('off');


# Okay.  We are ready to try to classify, right?
# 
# First we should check the proportions of each digit in the given data partitions.

# In[ ]:


classes = np.arange(10)
(Ttrain == classes).shape


# In[ ]:


(Ttrain == classes).sum(axis=0)


# In[ ]:


(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]


# In[ ]:


['Ttrain', *(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]]


# In[ ]:


import pandas

result = []
result.append(['Train', *(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]])
result.append(['Tval', *(Tval == classes).sum(axis=0) / Tval.shape[0]])
result.append(['Ttest', *(Ttest == classes).sum(axis=0) / Ttest.shape[0]])
pandas.DataFrame(result)


# All very close to 0.1. Super.

# Time for our first experiment.  Let's train a small neural net with 5 hidden units in one layer for a small number of epochs using Adam.

# In[ ]:


n_epochs = 100
learning_rate = 0.01

np.random.seed(142)

nnet = NeuralNetworkClassifier(Xtrain.shape[1], [5], len(classes))
nnet.train(Xtrain, Ttrain, n_epochs, learning_rate, method='adam', verbose=True)


# In[ ]:


print(nnet)  # uses the __str__ method


# In[ ]:


plt.plot(nnet.error_trace);


# Now it is time for you to run some longer experiments.  You must write the code to do the following steps:
# 
# 1. For each of at least five different hidden layer structures
# 
#     1. Train a network for 500 epochs.
#     1. Collect percent of samples correctly classified in the given train, validate, and test partitions.
# 
# 2. Create a `pandas.DataFrame` with these results and with column headings `('Hidden Layers', 'Train', 'Validate', 'Test', 'Time')` where `'Time'` is the number of seconds required to train each network.
# 
# 3. Retrain a network using the best hidden layer structure, judged by the percent correct on the validation set.
# 4. Use this network to find several images in the test set for which the network's probability of the correct class is the closest to zero, meaning images for which your network does the worst.  Draw these images and discuss why your network might not be doing well for those images.

# ## `confusion_matrix`
# 
# Now, write a function named `confusion_matrix` that returns a confusion matrix for any classification problem, returned as a `pandas.DataFrame` as shown in Lecture Notes 12.  It must require two arguments, the predicted classes for each sample and the true classes for each sample.  Here is an example.

# In[ ]:


Y_classes, Y_probs = nnet.use(Xtest)
confusion_matrix(Y_classes, Ttest)


# ## Grading and Check-In
# 
# You will receive 50 points for correct code, and 50 points for other results and your discussions.  As before, you can test your code against the grading script yourself by downloading [A4grader.zip](https://www.cs.colostate.edu/~cs445/notebooks/A4grader.zip) and extracting `A4grader.py` parallel to this notebook.  We recommend keeping this notebook and the grader script in a dedicated folder with *just those two files.* Run the code in the in the following cell to see an example grading run.  Submit assignments **through Canvas** following the pattern of the previous assignments. *Do not send your file to the instructor/TA via email or any other medium!*

# In[9]:


get_ipython().run_line_magic('run', '-i A4grader.py')


# ## Extra Credit
# Earn 5 extra credit point on this assignment by doing the following.
# 
# 1. Combine the train, validate, and test partitions loaded from the MNIST data file into two matrices, `X` and `T`. 
# 2. Using `adam` , `relu` and just one value of `learning_rate` and `n_epochs`, compare several hidden layer architectures. Do so by applying our `generate_k_fold_cross_validation_sets` function as defined in Lecture Notes 10 which forms stratified partitioning, for use in classification problems, to your `X` and `T` matrices using `n_fold` of 3.
# 3. Show results and discuss which architectures you find works the best, and how you determined this.

# In[ ]:




