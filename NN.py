import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tqdm
import copy
from utils import *

plt.style.use('ggplot')


class LinearLayer(Module):
    """
    A linear layer module which calculate (Wx + b).
    """

    def __init__(self, dim_in, dim_out, initializer, reg, alpha = 0):
        """
        Args:
            - dim_in: input dimension,
            - dim_out: output dimension,
            - initializer: a function which get (dim_in, dim_out) and initialize
                a [dim_in x dim_out] matrix,
            - reg: L2-regularization flag
            - alpha: L2-regularization coefficient
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params = {
            'W': initializer(self.dim_in, self.dim_out),
            'b': np.ones(self.dim_out),
            'reg' : reg,
            'alpha': alpha
        }
        self.grads = dict()
        self.cache = dict()

    def _forward(self, x):
        """
        linear forward function, calculate Wx+b for a batch of data

        Args:
            x : a batch of data

        """

        y = np.dot(x,self.params['W'].T) + self.params['b']
        self.cache['x'] = x
        self.cache['y'] = y
        return y

    def backward(self, upstream):
        """
        get upstream gradient and returns downstream gradient

        Args:
            upstream : upstream gradient of loss w.r.t module output
        """
        
        grad_b = np.sum(upstream, axis = 0)
        grad_w = np.dot(upstream.T, self.cache['x'])
        grad_x = np.dot(upstream, self.params['W'])
        grad_reg = {
            'W' : 2*self.params['W']*self.params['alpha'],
            'b' : 2*self.params['b']*self.params['alpha']
        }

        self.grads = {
            'W': grad_w,
            'b': grad_b,
            'x': grad_x,
            'reg': grad_reg
        }




class ReLU(Module):
    """
    Rectified Linear Unit function
    """

    def __init__(self):
        self.cache = dict()
        self.grads = dict()

    def _forward(self, x):
        """
        applies relu function on x

        Args:
            x : a batch of data

        Returns:
            y : relu of input
        """
        # print()

        # print(x)
        self.cache = {
            'x': x,
        }        
        return np.maximum(x, 0)

    def backward(self, upstream):
        """
        calculate and store gradient of loss w.r.t module input

        Args:
            upstream : gradient of loss w.r.t modele output
        """
        relu_func = np.vectorize(lambda x : np.maximum(x, 0)*1/x)
        grad_x = upstream * relu_func(self.cache['x'])        
        self.grads['x'] = grad_x


def logsumexp(array, axis=1):
    """
    calculate log(sum(exp(array))) using np.logaddexp

    Args:
        array : input array
        axis : reduce axis, 1 means columns and 0 means rows
    """
    assert len(array) >= 2
    return np.logaddexp.reduce(array, axis)



class LogSoftMax(Module):
    def __init__(self):
        self.cache = dict()
        self.grads = dict()

    def _forward(self, x):
        """
        get x and calculate softmax of that.

        Args:
            x : batch of data with shape (b,m)

        Returns:
            y : log softmax of x with shape (b,m)
        """
        
        y = x   
        index = 0
        for row in x:
            y[index] = (row - logsumexp(x, axis = 1)[index])
            index += 1
        self.cache['x'] = x
        self.cache['y'] = y
        return y

    def backward(self, upstream):
        """
        calculate gradient of loss w.r.t module input and save that in grads.

        Args:
            upstream : gradient of loss w.r.t module output with sahpe (b,m)
        """
        # grad_x = None

                
        # x = np.exp(self.cache['y'])   
        # index = 0
        # for row in x:
        #     y[index] = (row - logsumexp(x, axis = 1)[index])
        #     index += 1
        index = 0
        lst = np.exp(self.cache['y'])
        for row in np.exp(self.cache['y']):
            lst[index:] = row * upstream.sum(axis = 1)[index] - upstream[index]
            index += 1        
        grad_x = lst
        # print('upstream ', upstream.sum(axis=1))
        # print('y', self.cache['y'])
        # print('np.exp ', np.exp(self.cache['y']))
        # print('grad_x ', grad_x)
        # print('lst ', lst)
        # print("\n\n\n")
        self.grads['x'] = grad_x




class MLPModel(Module):
    """
    A multilayer neural network model
    """

    def __init__(self, layers):
        """
        Args:
            layers : list of model layers
        """
        self.layers = layers

    def _forward(self, x):
        """
        Perform forward on x

        Args:
            x : a batch of data

        Returns:
            o : model output
        """
        output = x
        for i in range(len(self.layers)):
            output = self.layers[i]._forward(output)
        return output
    
    def backward(self, upstream):
        """
        Perform backward path on whole model

        Args:
            upstream : gradient of loss w.r.t model output
        """
        
        # First we have to reverse the layers 
        layers = self.layers[::-1]
        for l in layers:
            l.backward(upstream)
            upstream = l.grads['x']



    def get_parameters(self):
        """
        Returns:
            parametric_layers : all layers of model which have parameter
        """
        res = []
        for layer in self.layers:
            try:
                if layer.params != None:
                    res.append(layer)
            except:
                pass
        return res
    



class CrossEntropyLoss(Module):
    def __init__(self, mean=False):
        self.mean = mean
        self.cache = dict()
        self.grads = dict()

    def _forward(self, logprobs, targets):
        """
        Calculate cross entropy of inputs.

        Args:
            probs : matrix of probabilities with shape (b,n)
            targets : list of samples classes with shape (b,)

        Returns:
            y : cross entropy loss
        """
        targ = np.zeros(logprobs.shape)
        y = np.zeros(targets.shape)
        for i in range(len(targets)):
            targ[i][targets[i]] = 1  
          
        # print(targ)
        self.cache = {
            'logprobs': logprobs,
            'targ' : targ,
            'len' : targ.shape[0]
        }
        # print("forward", targ)
        return -np.sum(targ * logprobs, axis=1) / targets.shape[0]

    def backward(self, upstream):
        """
        Calculate gradient of loss w.r.t module input and save them in grads.

        Args:
            upstream : gradient of loss w.r.t module output (loss)
        """
        grad = -upstream * self.cache['targ'] / self.cache['len']
        self.grads['x'] = grad    



class Optimizer():
    """
    """

    def __init__(self, layers, strategy, lr):
        """
        save layers here in order to update their parameters later.

        Args:
            layers : model layers (those that we want to update their parameters)
            strategy : optimization strategy
            lr : learning rate
        """
        self.layers = layers
        self.strategy = strategy
        self.lr = lr
        self.strategies = {
            'sgd': self._sgd,
            'momentum': self._momentum,
        }

    def step(self, *args):
        """
        Perform updating strategy on all layers paramters.
        """
        self.strategies[self.strategy](*args)

    def _sgd(self):
        """
        Perform sgd update on all parameters of layers
        """
        params = ['W', 'b']
        for layer in self.layers:
            try:
                for param in params:
                    layer.params[param] -= (layer.grads[param] + layer.grads['reg'][param]) * self.lr
            except:
                print("error occured in sgd")
    
    def _momentum(self):
        """
        Perform momentum update on all parameters of layers
        """
        self.changes = dict([])
        params = ['W', 'b']
        for i, layer in enumerate(self.layers):
            for param in params:
                try:
                    self.changes[(i, param)] = self.lr * (layer.grads[param] + layer.grads['reg'][param]) + self.strategies['momentum'] * self.changes.get((i, param), 0)
                    layer.params[param] -= self.lr * (layer.grads[param] + layer.grads['reg'][param]) + self.strategies['momentum'] * self.changes.get((i, param), 0)
                except:
                    print("error occured in momentum")