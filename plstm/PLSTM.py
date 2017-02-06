#!/usr/bin/env python
import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import L
import Mod

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
            W_ih=L.Linear(n_inputs, n_units),
            W_fh=L.Linear(n_inputs, n_units),
            W_ch=L.Linear(n_inputs, n_units),
            W_oh=L.Linear(n_inputs, n_units),
            W_ic=L.Linear(n_inputs, n_units),
            W_fc=L.Linear(n_inputs, n_units),
            W_ix=L.Linear(n_inputs, n_units),
            W_fx=L.Linear(n_inputs, n_units),
            W_ox=L.Linear(n_inputs, n_units),
            W_cx=L.Linear(n_inputs, n_units),
        )


class CoupledForgetLSTMBase(link.Chain):

    def __init__(self, n_units, n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(LSTMBase, self).__init__(
            W_fh=L.Linear(n_inputs, n_units),
            W_ch=L.Linear(n_inputs, n_units),
            W_oh=L.Linear(n_inputs, n_units),
            W_fx=L.Linear(n_inputs, n_units),
            W_cx=L.Linear(n_inputs, n_units),
            W_ox=L.Linear(n_inputs, n_units),
        )

class PeepHoleLSTMBase(link.Chain):

    def __init__(self, n_units, n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(PeepHoleLSTMBase, self).__init__(
            W_ih=L.Linear(n_inputs, n_units),
            W_fh=L.Linear(n_inputs, n_units),
            W_ch=L.Linear(n_inputs, n_units),
            W_oh=L.Linear(n_inputs, n_units),
            W_ic=L.Linear(n_inputs, n_units),
            W_fc=L.Linear(n_inputs, n_units),
            W_oc=L.Linear(n_inputs, n_units),
            W_ix=L.Linear(n_inputs, n_units),
            W_fx=L.Linear(n_inputs, n_units),
            W_ox=L.Linear(n_inputs, n_units),
            W_cx=L.Linear(n_inputs, n_units),
        )

class CoupledForgetPeepHoleLSTMBase(link.Chain):

    def __init__(self, n_units, n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(PeepHoleLSTMBase, self).__init__(
            W_fh=L.Linear(n_inputs, n_units),
            W_fc=L.Linear(n_inputs, n_units),
            W_oh=L.Linear(n_inputs, n_units),
            W_oc=L.Linear(n_inputs, n_units),
            W_ch=L.Linear(n_inputs, n_units),
            W_fx=L.Linear(n_inputs, n_units),
            W_ox=L.Linear(n_inputs, n_units),
            W_cx=L.Linear(n_inputs, n_units),
        )

class PhasedLSTM(PLSTMBase):
    def __init__(self, in_size, out_size,
                 Period=Variable(np.array([50]).astype(np.float32)),
                 Shift=Variable(np.array([500]).astype(np.float32)),
                 On_End=Variable(np.array([0.005]).astype(np.float32)),
                 Alfa=Variable(np.array([0.0001]).astype(np.float32))):
        super(PhasedLSTM, self).__init__(out_size,in_size)
        self.t=Variable(np.array([0.]).astype(np.float32))
        self.reset_state()
        self.Period = Period
        self.Shift = Shift
        self.On_End = On_End
        self.Alfa=Alfa
    def __call__(self,x,h,c):
        phai=Mod.mod((self.t-self.Shift),self.Period)/self.Period
        k=F.where(phai<self.On_End/2 , 2*phai/self.On_End , 2-2*phai/self.On_End)
        k=F.where(phai>self.On_End,self.Alfa*phai,k,k)

        ft = sigmoid.sigmoid(self.W_fx(x) + self.W_fh(h) + self.W_fc(c))
        it = sigmoid.sigmoid(self.W_ix(x) + self.W_ih(h) + self.W_ic(c))
        ct = tanh.tanh(self.W_cx(x) + self.W_ch(h))
        ct = ft * c + it * ct
        c  = k*ct +(1-k)*c
        ot = sigmoid.sigmoid(self.W_ox(x) + self.W_oh(h) + self.W_oc(c))
        ht = ot * tanh.tanh(c)
        h =  k*ht +(1-k)*h
        return h, c
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

class StatefulPeepHoleLSTM(PeepHoleLSTMBase):


    def __init__(self, in_size, out_size):
        super(StatefulPeepHoleLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulPeepHoleLSTM, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()
        if self.c is not None:
            self.c.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulPeepHoleLSTM, self).to_gpu(device)
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

        if self.h is not None and self.c is not None:
            ft += self.W_fh(h) + self.W_fc(self.c)
            it += self.W_ih(h) + self.W_ic(self.c)
            ct += self.W_ch(h)
            ot += self.W_oh(h)
        ft = sigmoid.sigmoid(ft)
        it = sigmoid.sigmoid(it)
        ct = tanh.tanh(ct)
        ot = sigmoid.sigmoid(ot + self.W_oc(ct))

        c = it * ct
        if self.c is not none:
            self.c += ft * c

        self.h = ot * tanh.tanh(self.c)
        return self.h

    def get_state():
        return self.c


class StatelessPeepHoleLSTM(PeepHoleLSTMBase):


    def __init__(self, in_size, out_size):
        super(StatelessPeepHoleLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size


    def __call__(self, x, h, c):
        ft = sigmoid.sigmoid(self.W_fx(x) + self.W_fh(h) + self.W_fc(c))
        it = sigmoid.sigmoid(self.W_ix(x) + self.W_ih(h) + self.W_ic(c))
        ct = tanh.tanh(self.W_cx(x) + self.W_ch(h))
        c = ft * c + it * ct
        ot = sigmoid.sigmoid(self.W_ox(x) + self.W_oh(h) + self.W_oc(c))
        h = ot * tanh.tanh(c)
        return h, c

class CoupledForgetStatefulLSTM(CoupledForgetLSTMBase):


    def __init__(self, in_size, out_size):
        super(CoupledForgetStatefulLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(CoupledForgetStatefulLSTM, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()
        if self.c is not None:
            self.c.to_cpu()

    def to_gpu(self, device=None):
        super(CoupledForgetStatefulLSTM, self).to_gpu(device)
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
        ct = self.W_cx(x)
        ot = self.W_ox(x)

        if self.h is not None:
            ft += self.W_fh(h)
            ct += self.W_ch(h)
            ot += self.W_oh(h)
        ft = sigmoid.sigmoid(ft)
        ct = tanh.tanh(ct)
        ot = sigmoid.sigmoid(ot)

        c = (1 - ft) * ct
        if self.c is not none:
            c += ft * self.c
        self.c = c
        self.h = ot * tanh.tanh(self.c)
        return self.h

    def get_state():
        return self.c


class CoupledForgetStatelessLSTM(CoupledForgetLSTMBase):
    def __init__(self, in_size, out_size):
        super(CoupledForgetStatelessLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size

    def __call__(self, x, h, c):
        ft = sigmoid.sigmoid(self.W_fx(x) + self.W_fh(h))
        ct = tanh.tanh(self.W_cx(x) + self.W_ch(h))
        ot = sigmoid.sigmoid(self.W_ox(x) + self.W_oh(h))
        c = ft * c + (1 - ft)) * ct
        h = ot * tanh.tanh(c)
        return h, c

class CoupledForgetStatefulPeepHoleLSTM(CoupledForgetPeepHoleLSTMBase):


    def __init__(self, in_size, out_size):
        super(CoupledForgetStatefulPeepHoleLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(CoupledForgetStatefulPeepHoleLSTM, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()
        if self.c is not None:
            self.c.to_cpu()

    def to_gpu(self, device=None):
        super(CoupledForgetStatefulPeepHoleLSTM, self).to_gpu(device)
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
        ct = self.W_cx(x)
        ot = self.W_ox(x)

        if self.h is not None and self.c is not None:
            ft += self.W_fh(h) + self.W_fc(self.c)
            ct += self.W_ch(h)
            ot += self.W_oh(h)
        ft = sigmoid.sigmoid(ft)
        ct = tanh.tanh(ct)
        ot = sigmoid.sigmoid(ot + self.W_oc(ct))

        c = (1 - ft) * ct
        if self.c is not none:
            self.c += ft * c

        self.h = ot * tanh.tanh(self.c)
        return self.h

    def get_state():
        return self.c


class CoupledForgetStatelessPeepHoleLSTM(CoupledForgetPeepHoleLSTMBase):


    def __init__(self, in_size, out_size):
        super(CoupledForgetStatelessPeepHoleLSTM, self).__init__(out_size, in_size)
        self.state_size = out_size


    def __call__(self, x, h, c):
        ft = sigmoid.sigmoid(self.W_fx(x) + self.W_fh(h) + self.W_fc(c))
        ct = tanh.tanh(self.W_cx(x) + self.W_ch(h))
        c = ft * c + (1 - ft) * ct
        ot = sigmoid.sigmoid(self.W_ox(x) + self.W_oh(h) + self.W_oc(c))
        h = ot * tanh.tanh(c)
        return h, c
