#!/usr/bin/env python
import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link,Variable
import chainer.links as L
import Mod
import numpy as np
#from chainer import cuda

import chainer.functions as F
import os
import init
#os.environ["CHAINER_TYPE_CHECK"] = "0"

class LSTMBase(link.Chain):

    def __init__(self, n_units, n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(LSTMBase, self).__init__(
            W_ih=L.Linear(n_inputs, n_units),
            W_fh=L.Linear(n_inputs, n_units),
            W_ch=L.Linear(n_inputs, n_units),
            W_oh=L.Linear(n_inputs, n_units),
            W_ix=L.Linear(n_inputs, n_units),
            W_fx=L.Linear(n_inputs, n_units),
            W_cx=L.Linear(n_inputs, n_units),
            W_ox=L.Linear(n_inputs, n_units),
        )

class PLSTMBase(chainer.Chain):
    def __init__(self,n_units,n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(PLSTMBase, self).__init__(
            W_ih=L.Linear(n_units, n_units),
            W_fh=L.Linear(n_units, n_units),
            W_ch=L.Linear(n_units, n_units),
            W_oh=L.Linear(n_units, n_units),
            W_ix=L.Linear(n_inputs, n_units),
            W_fx=L.Linear(n_inputs, n_units),
            W_cx=L.Linear(n_inputs, n_units),
            W_ox=L.Linear(n_inputs, n_units),
        )

class PhasedLSTM(PLSTMBase):
    def __init__(self, in_size, out_size,
                 Period=init.Uniform((10,100)),
                 Shift=init.Uniform( (0., 1000.)),
                 On_End=init.Constant(0.05),
                 Alfa=init.Constant(0.001)):
        super(PhasedLSTM, self).__init__(out_size,in_size)
        self.reset_state()
        self.t=0
        self.reset_state()
        self.Period = Period.sample((20,out_size))
        self.Shift = Shift.sample((20,out_size))
        self.On_End = On_End.sample((20,out_size))
        self.Alfa=Alfa.sample((20,out_size))
    def __call__(self,x):
        #import pdb; pdb.set_trace()
        phi=Mod.mod((self.t-self.Shift),self.Period)/self.Period
        k=F.where(phi.data<self.On_End/2 , 2*phi/self.On_End , 2-2*phi/self.On_End)
        k=F.where(phi.data>self.On_End,self.Alfa*phi,k)

        ft = self.W_fx(x)
        it = self.W_ix(x)
        ct = self.W_cx(x)
        ot = self.W_ox(x)

        if self.h is not None:
            ft += self.W_fh(self.h)
            it += self.W_ih(self.h)
            ct += self.W_ch(self.h)
            ot += self.W_oh(self.h)
        ft = sigmoid.sigmoid(ft)
        it = sigmoid.sigmoid(it)
        ct = tanh.tanh(ct)
        ot = sigmoid.sigmoid(ot)

        self.ct = it * ct
        if self.c is not None:
            self.ct += ft * self.c
        c=k*ct
        if self.c is not None:
            self.c += (1-k)*self.c
        self.c=c

        ht = ot * tanh.tanh(ct)
        h=k*ht
        if self.h is not None:
            h+=(1-k)*self.h
        self.h=h
        return self.h


    def set_state(self, h, c):
        assert isinstance(h, chainer.Variable)
        assert isinstance(c, chainer.Variable)
        h_ = h
        c_ = c
        self.h = h_
        self.c = c_

    def reset_state(self):
        self.h = None
        self.c = None
        self.t = 0


class StatefulLSTM(LSTMBase):


    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulLSTM, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()
        if self.c is not None:
            self.c.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h, c):
        assert isinstance(h, chainer.Variable)
        assert isinstance(c, chainer.Variable)
        h_ = h
        c_ = c
        if self.xp == numpy:
            h_.to_cpu()
            c_.to_cpu()
        else:
            h_.to_gpu()
            c_.to_gpu()
        self.h = h_
        self.c = c_

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, x):
        ft = self.W_fx(x)
        it = self.W_ix(x)
        ct = self.W_cx(x)
        ot = self.W_ox(x)

        if self.h is not None:
            ft += self.W_fh(h)
            it += self.W_ih(h)
            ct += self.W_ch(h)
            ot += self.W_oh(h)
        ft = sigmoid.sigmoid(ft)
        it = sigmoid.sigmoid(it)
        ct = tanh.tanh(ct)
        ot = sigmoid.sigmoid(ot)

        c = it * ct
        if self.c is not none:
            c += ft * self.c
        self.c = c
        self.h = ot * tanh.tanh(self.c)
        return self.h

    def get_state():
        return self.c


class StatelessLSTM(LSTMBase):
    def __init__(self, in_size, out_size):
        super(StatelessLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size

    def __call__(self, x, h, c):
        ft = sigmoid.sigmoid(self.W_fx(x) + self.W_fh(h))
        it = sigmoid.sigmoid(self.W_ix(x) + self.W_ih(h))
        ct = tanh.tanh(self.W_cx(x) + self.W_ch(h))
        ot = sigmoid.sigmoid(self.W_ox(x) + self.W_oh(h))
        c = ft * c + it * ct
        h = ot * tanh.tanh(c)
        return h, c
