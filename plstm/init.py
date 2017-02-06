import numpy as np
from chainer import cuda
#import cuda.cupy as cuda.cupy
xp=cuda.cupy
class Initializer(object):
    """Base class for parameter tensor initializers.
    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.
    """
    def __call__(self, shape):
        """
        Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`sample()` method.
        """
        return self.sample(shape)

    def sample(self, shape):
        """
        Sample should return a theano.tensor of size shape and data type
        theano.config.floatX.
        Parameters
        -----------
        shape : tuple or int
            Integer or tuple specifying the size of the returned
            matrix.
        returns : theano.tensor
            Matrix of size shape and dtype theano.config.floatX.
        """
        raise NotImplementedError()

    def to_gpu(self):
        global xp
        xp = cuda.cupy

class Constant(Initializer):
    """Initialize weights with constant value.
    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, val=0.0):
        self.val = val
        cuda.get_device(1).use()
    def sample(self, shape):
        return xp.array(xp.ones(shape) * self.val).astype(xp.float32)
    def to_gpu(self):
        global xp
        xp = cuda.cupy


class Uniform(Initializer):
    """Sample initial weights from the uniform distribution.
    Parameters are sampled from U(a, b).
    Parameters
    ----------
    range : float or tuple
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    std : float or None
        If std is a float then the weights are sampled from
        U(mean - cuda.cupy.sqrt(3) * std, mean + cuda.cupy.sqrt(3) * std).
    mean : float
        see std for description.
    """
    def __init__(self, range=0.01, std=None, mean=0.0):
        if std is not None:
            a = mean - xp.sqrt(3) * std
            b = mean + xp.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number

        self.range = (a, b)
        cuda.get_device(1).use()

    def sample(self, shape):
        #print("return numpy array")
        return xp.array((xp.random.uniform(low=self.range[0], high=self.range[1], size=shape))).astype(xp.float32)

    def to_gpu(self):
        global xp
        xp = cuda.cupy
