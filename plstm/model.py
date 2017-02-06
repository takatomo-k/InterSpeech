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



class BaseDecoder(chainer.Chain):
    def __init__(self, x, hidden_units, depth=1, drop_ratio=0.):
        super(BaseDecoder, self).__init__(
                dec=StackLSTM(x, hidden_units, depth, drop_ratio)
                #dec=PhasedLSTM(x,hidden_units)
             )
    def reset(self):
        self.dec.reset_state()

    def __call__(self, x):
        return self.dec(x)


class BaseEncoder(chainer.Chain):
    def __init__(self, x, hidden_units, depth=1, drop_ratio=0.):
        super(BaseEncoder, self).__init__(

                encF=L.StatefulPeepholeLSTM(x, hidden_units),
                encB=L.StatefulPeepholeLSTM(x, hidden_units),
                #encF=PhasedLSTM(x,hidden_units),
                #encB=PhasedLSTM(x,hidden_units),
                aw = L.Linear(hidden_units * 2, hidden_units)
                )
        self._train = True

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
