from __future__ import print_function
import sys
#sys.path.append('/home/is/takatomo-k/work/End-to-End/lstm/att_mt_emph_trans/create_model/')
import chainer
import component as com
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
import model

class Create_Model(chainer.Chain):
    def __init__(self, input_units, hidden_unit, output_units, depth=1, \
            drop_ratio=0., src_vocab=None, trg_vocab=None, att_type="dot"):
        print('~~~~~~~ Model parameters ~~~')
        print('~ Input size:', input_units)
        print('~ Output size:', output_units)
        print('~ Hidden size:', hidden_unit)
        print('~ Attention  :', att_type)
        print('~ Layer depth:', depth)
        print('~ Drop ratio:', drop_ratio)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        self.input_units = input_units
        self.hidden_unit = hidden_unit
        self.output_units = output_units
        self.depth = depth
        self.drop_ratio = drop_ratio
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        enc = Custom_Encoder(input_units, hidden_unit, depth, drop_ratio)
        dec = Custom_Decoder(hidden_unit, output_units, depth, drop_ratio)
        att = Custom_Attention(hidden_unit, att_type=att_type)
        e2h = L.Linear(output_units, hidden_unit)
        super(Create_Model, self).__init__(enc=enc, att=att, dec=dec, e2h=e2h)
        self.S = None
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.enc.train = value
        self.dec.train = value

    def to_gpu(self):
        global xp
        xp = cuda.cupy
        super(Create_Model, self).to_gpu()

    def encode(self, x_batch):
        #import pdb; pdb.set_trace()
        batch_size = x_batch[0].data.shape[0]
        self.S, e_l = self.enc(x_batch)
        self.dec.reset(e_l)
    def reset(self):
        self.S = None
        self.enc.reset()
        self.dec.reset()
        self.att.reset()

    def decode(self, y, no_att=False):
        h = self.dec.update(self.e2h(y))

        if no_att:
            o = self.dec()
        else:
            att = self.att(self.S, h)
            o = self.dec(att_in=att)
        return o


class Custom_Encoder(model.BaseEncoder):
    def __init__(self, isize, osize, depth=1, drop_ratio=0.):
        super(Custom_Encoder, self).__init__(isize, osize, depth, drop_ratio)
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        super(Custom_Encoder, self).train(value)

    def __call__(self, x_list):
        batch_size = x_list[0].data.shape[0]
        e_list = super(Custom_Encoder, self).encode(x_list)
        #import pdb; pdb.set_trace()
        S = F.reshape(F.concat(e_list, axis=1), (batch_size, x_list.shape[0], -1))
        return S, e_list[-1]

    def reset(self):
        super(Custom_Encoder, self).reset()


class Custom_Decoder(model.BaseDecoder):
    def __init__(self, isize, osize, depth=1, drop_ratio=0.):
        super(Custom_Decoder, self).__init__(isize, isize, depth, drop_ratio)
        self.add_link('wc', L.Linear(isize * 2, isize))
        self.add_link('wo', L.Linear(isize, osize))
        self.add_link('nc',L.LayerNormalization(isize))
        self.add_link('no',L.LayerNormalization(osize))
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        super(Custom_Decoder, self).train(value)

    def update(self, x):
        self.h = super(Custom_Decoder, self).__call__(x)
        return self.h

    def __call__(self, att_in=None, output_hidden=False):
        if att_in is not None:
            h = F.tanh(self.nc(self.wc(F.concat((self.h, att_in), axis=1))))
        else:
            h = self.h
        self.o = self.no(self.wo(h))
        if output_hidden:
            return self.o, h
        else:
            return self.o

    def reset(self, e_l=None):
        super(Custom_Decoder, self).reset()
        if e_l is not None:
           self.start_with(e_l)

    def start_with(self, x):
        o = super(Custom_Decoder, self).__call__(x)


class Custom_Attention(model.BaseAttention):
    def __init__(self, osize, att_type="dot"):
        super(Custom_Attention, self).__init__(osize, att_type=att_type)

    def __call__(self, enc_mat, h):
        return super(Custom_Attention, self).__call__(enc_mat, h)
