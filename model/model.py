#!/usr/bin/env python
from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, cuda,optimizer, optimizers, serializers,Variable
from chainer.training import extensions
import numpy as np
import sys
from chainer import ChainList
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.utils import type_check
from chainer import utils

class AttentionS2S(chainer.Chain):
  def __init__(self,src_dim,hidden_units,tgt_dim):
    super(AttentionS2S, self).__init__(
        fenc = Encoder(src_dim,hidden_units),
        benc = Encoder(src_dim,hidden_units),
        att = Attention(hidden_units),
        dec = Decoder(hidden_units,tgt_dim),
    )
    self.hidden_units = hidden_units
    self.tgt_dim = tgt_dim
    self.src_dim=src_dim
  def reset(self,batchsize):
    self.zerograds()
    self.x_list = []

  def append(self, x):
      self.x_list.append(x)

  def encode(self):
    batchsize =self.x_list[0].shape[0]
    ZEROS = Variable(xp.zeros((batchsize, self.hidden_units)).astype(xp.float32))
    c = ZEROS
    a = ZEROS
    a_list = []
    for x in self.x_list:
      c, a = self.fenc(x, c, a)
      a_list.append(a)
    c = ZEROS
    b = ZEROS
    b_list = []
    for x in reversed(self.x_list):
      c, b = self.benc(x, c, b)
      b_list.insert(0, b)
    self.a_list = a_list
    self.b_list = b_list
    self.c = ZEROS
    self.h = ZEROS

  def decode(self, y):

      aa, bb = self.att(self.a_list, self.b_list, self.h)
      y, self.c, self.h = self.dec(y, self.c, self.h, aa, bb)
      return y

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.hidden_units, file=fp)
      print(self.tgt_dim, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      src_dim = int(next(fp))
      hidden_units = int(next(fp))
      tgt_dim= int(next(fp))
      return AttentionS2S(src_dim,hidden_units,tgt_dim)


class StackLSTM(ChainList):
    def __init__(self, x, O, depth=1, drop_ratio=0):
        chain_list = []
        for i in range(depth):
            start = x if i == 0 else O
            #import pdb; pdb.set_trace()
            #chain_list.append(PLSTMBase(start, O))
            chain_list.append(L.LSTM(start,O))
        
        self._drop_ratio = drop_ratio
        super(StackLSTM, self).__init__(*chain_list)
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, flag):
        self._train = flag

    def reset_state(self):
        for lstm in self:
            lstm.reset_state()

    def __call__(self, x, is_train=False):
        ret = None
        for i, hidden_units in enumerate(self):
            h = x if i == 0 else ret
            ret = hidden_units(h)
        if self.train and self._drop_ratio:
            return F.dropout(ret, train=is_train, ratio=self._drop_ratio)
        else:
            return ret

    def get_state(self):
        ret = []
        for lstm in self:
            ret.append((lstm.c, lstm.h))
        return ret

    @property
    def h(self):
        return self.get_state()[-1][-1]

    @h.setter
    def h(self, h):
        for lstm_self in self:
            lstm_self.h = h

    def set_state(self, state):
        for lstm_self, lstm_in in zip(self, state):
            lstm_self.c, lstm_self.h = lstm_in


class BaseAttention(chainer.Chain):
    def __init__(self, hidden_units, att_type="dot"):
        self.WA = None
        if att_type == "general":

            self.WA = L.Linear(hidden_units, hidden_units)
        elif att_type == "concat":

            self.WA = L.Linear(hidden_units * 2, hidden_units)

        if att_type == "dot":

            super(BaseAttention, self).__init__()
        else:

            super(BaseAttention, self).__init__(WA=self.WA)
        self.hidden_units = hidden_units
        self.att_type = att_type

    def __call__(self, enc_mat, h):
        if self.att_type == "dot":
            return self._dot(enc_mat, h)
        elif self.att_type == "general":
            return self._general(enc_mat, h)
        elif self.att_type == "concat":
            return self._concat(enc_mat, h)
        else:
            print("Attentional type", self.att_type, "is not supported", file=sys.stderr)
            exit(1)

    def _dot(self, enc_mat, h):
        weights = F.softmax(F.batch_matmul(enc_mat, h))
        att = F.reshape(F.batch_matmul(weights, enc_mat, transa=True), (h.data.shape[0], self.hidden_units))
        return att

    def _general(self, enc_mat, h):
        batch, src_len, hidden = enc_mat.data.shape
        param_s = F.reshape(self.WA(F.reshape(enc_mat, (batch * src_len, hidden))), (batch, src_len, hidden))
        return self._dot(param_s, h)

    def _concat(self, enc_mat, h):
        batch, src_len, hidden = enc_mat.data.shape
        concat_h  = F.reshape(F.concat(F.broadcast(F.expand_dims(h, 1), enc_mat), axis=1), (batch * src_len, 2* hidden))
        return F.softmax(F.reshape(self.WA(concat_h), (batch, src_len)))

    def reset(self):
        pass

class PLSTMBase(chainer.Chain):
    def __init__(self,n_units,n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(PLSTMBase, self).__init__(
            W_fh=L.Linear(n_inputs, n_units),
            W_ih=L.Linear(n_inputs, n_units),
            W_oh=L.Linear(n_inputs, n_units),
            W_ch=L.Linear(n_inputs, n_units),
            W_fx=L.Linear(n_inputs, n_units),
            W_ix=L.Linear(n_inputs, n_units),
            W_ox=L.Linear(n_inputs, n_units),
            W_cx=L.Linear(n_inputs, n_units),
        )




class Mod(chainer.function.Function) :

    @property
    def label(self) :
        return '__mod__'

    def check_type_forward(self, in_types) :
        type_check.expect(in_types.size() == 2)
        #import pdb; pdb.set_trace()
        type_check.expect(in_types[0].dtype.kind == in_types[1].dtype.kind,in_types[0].shape == in_types[1].shape)

    def forward(self, x) :
        return utils.force_array(x[0] % x[1]),

    def backward(self, x, gy) :
        return gy[0], -(x[0] // x[1])*gy[0]

def mod(x, y) :
    return Mod()(x, y)

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
    def reset_state(self):
        self.h = None
        self.c = None
    def __call__(self,x):

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
        phai=mod((self.t-self.Shift),self.Period)/self.Period
        k=F.where(phai<self.On_End/2 , 2*phai/self.On_End , 2-2*phai/self.On_End)
        k=F.where(phai>self.On_End,self.Alfa*phai,k)

        c = it * ct
        if self.c is not None:
            c += ft * self.c
        pc = k*c
        if self.c is not None:
            pc+=(1-k)*self.c

        self.c = pc

        h = ot * tanh.tanh(self.c)
        ph = k*h
        if self.h is not None:
            ph+=(1-k)*self.h
        self.h=ph

        return self.h

class BaseDecoder(chainer.Chain):
    def __init__(self, x, hidden_units, depth=1, drop_ratio=0.):
        super(BaseDecoder, self).__init__(
                dec=StackLSTM(x, hidden_units, depth, drop_ratio)
                #dec=PhasedLSTM(x,hidden_units)
             )
        self._train = True

    def train(self, flag):
        self._train = flag
        self.dec.train = flag

    def reset(self):
        self.dec.reset_state()

    def __call__(self, x):
        return self.dec(x)


class BaseEncoder(chainer.Chain):
    def __init__(self, x, hidden_units, depth=1, drop_ratio=0.):
        super(BaseEncoder, self).__init__(

                encF=StackLSTM(x, hidden_units, depth, drop_ratio),
                encB=StackLSTM(x, hidden_units, depth, drop_ratio),
                #encF=PhasedLSTM(x,hidden_units),
                #encB=PhasedLSTM(x,hidden_units),
                aw = L.Linear(hidden_units * 2, hidden_units)
                )
        self._train = True

    def train(self, flag):
        self._train = flag
        self.encF.train = flag
        self.encB.train = flag

    def _encode_forward(self, x):
        return self.encF(x)

    def _encode_backward(self, x):
        return self.encB(x)

    def encode(self, x_list):
        self.reset()
        fx_list = []  # forward encoded list
        bx_list = []  # backward encoded list

        for idx, x in enumerate(x_list):
            fx = self._encode_forward(x)   # encode forward
            bx = self._encode_backward(x_list[-idx - 1])  # encode backward
            fx_list.append(fx)
            bx_list.append(bx)

        e_list = []
        #import pdb; pdb.set_trace()
        for idx in range(x_list.shape[0]):
            fx_i = fx_list[idx]
            bx_i = bx_list[-idx - 1]
            e_i = self.aw(F.concat((fx_i, bx_i), axis=1))
            e_list.append(e_i)
        return e_list

    def reset(self):
        self.encF.reset_state()
        self.encB.reset_state()
