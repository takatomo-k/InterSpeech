#!/usr/bin/env python
from __future__ import print_function
import argparse
import glob
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, cuda,optimizer, optimizers, serializers,Variable
from chainer.training import extensions
from sklearn.cross_validation import train_test_split
import numpy as np
import sys
import datetime

# Network definition
train_files = glob.glob('./corpus/es/*')
test_files = glob.glob('./corpus/en/*')
xp =cuda.cupy

src_max_len=800
tgt_max_len=800

class MLP(chainer.Chain):

    def __init__(self, hidden_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, hidden_units),  # n_in -> hidden_units
            l2=L.Linear(None, hidden_units),  # hidden_units -> hidden_units
            l3=L.Linear(None, n_out),  # hidden_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class Classifier(chainer.Chain):
    def __init__(self,predictor):
        super(Classifier,self).__init__(predictor=predictor)

    def __call__(self,x,t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y,t)
        accuracy = F.accuracy(y,t)

        return loss

class Attention(chainer.Chain):
    def __init__(self,hidden_units):
        super(Attention,self).__init__(
            l1=L.Linear(hidden_units,hidden_units),
            l2=L.Linear(hidden_units,hidden_units),
            l3=L.Linear(hidden_units,hidden_units),
            l4=L.Linear(hidden_units,1),
        )
        self.hidden_units = hidden_units

    def __call__(self, a_list, b_list, p):

        batchsize = p.data.shape[0]
        e_list = []
        sum_e = xp.zeros((batchsize, 1)).astype(xp.float32)
        for a, b in zip(a_list, b_list):
            w = F.tanh(self.l1(a) + self.l2(b) + self.l3(p))
            e = F.exp(self.l4(w))
            e_list.append(e)
            #import pdb; pdb.set_trace()

            sum_e =sum_e + e


        ZEROS = Variable(xp.zeros((batchsize, self.hidden_units)).astype(xp.float32))
        aa = ZEROS
        bb = ZEROS
        for a, b, e in zip(a_list, b_list, e_list):
            #import pdb; pdb.set_trace()
            e = e/sum_e
            aa += F.reshape(F.batch_matmul(a, e), (batchsize, self.hidden_units))
            bb += F.reshape(F.batch_matmul(b, e), (batchsize, self.hidden_units))

        return aa, bb

class Decoder(chainer.Chain):
  def __init__(self, hidden_units,tgt_dim):
    super(Decoder, self).__init__(
        #ye = links.EmbedID(vocab_size, embed_size),
        l1=L.LSTM(tgt_dim,4*hidden_units),
        l2=L.Linear(hidden_units, 4 * hidden_units),
        l3=L.Linear(hidden_units, 4 * hidden_units),
        l4=L.Linear(hidden_units, 4 * hidden_units),
        l5=L.Linear(hidden_units, tgt_dim),

    )

  def __call__(self, y, c, h, a, b):
    #import pdb; pdb.set_trace()

    c, h = F.lstm(c, self.l1(y) + self.l2(h) + self.l3(a) + self.l4(b))

    return self.l5(h), c, h




class Encoder(chainer.Chain):
    def __init__(self,in_units,hidden_units):
        super(Encoder,self).__init__(
        l1=L.Linear(in_units,4*hidden_units),
        l2=L.Linear(hidden_units,4*hidden_units),
        )

    def reset(self):
        self.zerograds()

    def __call__(self,x,c,h):
        #print(x.dtype,c.dtype,h.dtype)
        #print(x.shape)
        #import pdb; pdb.set_trace()
        tmp=self.l1(x)
        tmp2=self.l2(h)
        return F.lstm(c,tmp+tmp2)

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
      #import pdb; pdb.set_trace()
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


def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()

def forward(src_batch, trg_batch, atts2s, is_training, generation_limit):
  batchsize = len(src_batch)
  src_len = len(src_batch[0])
  trg_len = len(trg_batch[0]) if trg_batch else 0
  atts2s.reset(batchsize)

  for x in src_batch:
      atts2s.append(x)

  atts2s.encode()


  hyp_batch = [[] for _ in range(batchsize)]

  if is_training:
    loss = xp.zeros(())
    for t in trg_batch:
        y = atts2s.decode(t)
        #import pdb; pdb.set_trace()
        loss = loss +  F.mean_squared_error(y, t)
        output = cuda.to_cpu(y.data.argmax(1))
        for k in range(batchsize):
            hyp_batch[k].append(output[k])
    return hyp_batch, loss

  else:
    while len(hyp_batch[0]) < generation_limit:
      y = atts2s.decode(t)
      output = cuda.to_cpu(y.data.argmax(1))
      t = xp.iarray(output)
      for k in range(batchsize):
        hyp_batch[k].append(output[k])
      if all(hyp_batch[k][-1] == pad for k in range(batchsize)):
        break

    return hyp_batch

def train(args):
  #trace('making vocabularies ...')
  #src_vocab = Vocabulary.new(gens.word_list(args.source), args.vocab)
  #trg_vocab = Vocabulary.new(gens.word_list(args.target), args.vocab)

  trace('making model ...')
  atts2s = AttentionS2S(13,args.hidden_units,25)
  if args.gpu:
      atts2s.to_gpu()

  #batch_list=create_batch(train_files,test_files,args.batchsize);

  for epoch in range(args.epoch):
    trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
    trained = 0
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(atts2s)
    opt.add_hook(optimizer.GradientClipping(5))

    count=0
    for src_batch, trg_batch in create_batch(train_files,test_files,args.batchsize):
      count+=1
      K = len(src_batch)
      hyp_batch, loss = forward(src_batch, trg_batch, atts2s, True, 1600)
      if count%5 ==0 or count ==K:
          print("reset")
          atts2s.reset(src_batch)
          loss.backward()
          loss.unchain_backward()
          opt.update()

      for k in range(K):
        trace('epoch %3d/%3d, sample  %3f' % (epoch + 1, args.epoch, loss.data))

      trained += K

    trace('saving model ...')
    prefix = args.model + '.%03.d' % (epoch + 1)
    atts2s.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', atts2s)
    #import pdb; pdb.set_trace()

  trace('finished.')

def test(args):
  trace('loading model ...')
  atts2s = AttentionS2S.load_spec(args.model + '.spec')
  if args.gpu:
    atts2s.to_gpu()
  serializers.load_hdf5(args.model + '.weights', atts2s)

  trace('generating translation ...')
  generated = 0

  with open(args.target, 'w') as fp:
    for src_batch in gens.batch(gens.word_list(args.source), args.minibatch):
      src_batch = fill_batch(src_batch)
      K = len(src_batch)

      trace('sample %8d - %8d ...' % (generated + 1, generated + K))
      hyp_batch = forward(src_batch, None, src_vocab, trg_vocab, atts2s, False, args.generation_limit)

      for hyp in hyp_batch:
        hyp.append('</s>')
        hyp = hyp[:hyp.index('</s>')]
        print(' '.join(hyp), file=fp)

      generated += K

  trace('finished.')

def create_batch(train,target, batchsize):
    train_batch=[]
    target_batch=[]

    max_len=0
    batch_num=0

    for train_file, test_file in zip(train,target):

        x =np.genfromtxt(train_file, delimiter=" ").astype(xp.float32)
        y =np.genfromtxt(test_file, delimiter=" ").astype(xp.float32)
        source=xp.zeros((src_max_len,x.shape[1])).astype(xp.float32)
        test=xp.zeros((tgt_max_len,y.shape[1])).astype(xp.float32)
        #import pdb; pdb.set_trace()
        #print(test.shape)
        if(len(x)>src_max_len or len(y)>src_max_len):
            continue

        for i in range(len(x)):
            for j in range(len(x[i])):
                source[i][j]=x[i][j].astype(xp.float32)
        for k in range(len(y)):
            for m in range(len(y[k])):
                test[k][m]=y[k][m].astype(xp.float32)

        if(len(x)>max_len):
            max_len=len(x)
            train_batch.insert(0,source)
            target_batch.insert(0,test)
        else:
            train_batch.append(source)
            target_batch.append(test)
        batch_num+=1
        if(batch_num>=batchsize):
            #import pdb; pdb.set_trace()
            yield train_batch,target_batch
            train_batch=[]
            target_batch=[]
            batch_num=0

    yield train_batch,target_batch

def main():
    parser = argparse.ArgumentParser(description='Chainer example')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--hidden_units', '-hu  ', type=int, default=512,
                        help='Number of hidden_units')
    parser.add_argument('--mode',type=str,default='train',help='\'train\' or \'test\'')
    parser.add_argument('--model', type=str,default='in', help='[in/out] model file')
    args = parser.parse_args()


    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))

    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    if(args.gpu >= 0):
        cuda.check_cuda_available()
        if(args.gpu >= 0) :
            xp = cuda.cupy
            print ("use GPU")
        else :
            np
            print("use cpu")

    #xp.set_library(args)
    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.


    #model = FanctionSet(l1=F.Linear(784,hidden_units),
    #                    l2=F.Linear(hidden_units,hidden_units),
    #                    l3=F.Linear(hidden_units,10))
    #if args.gpu >= 0:
    #    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    #    model.to_gpu()  # Copy the model to the GPU


    train(args)

    # Load the MNIST dataset
    #train, test = chainer.datasets.get_mnist()
    #train, test = Data()
    #model = LSTM(args.unit, len(test[0]))
    #train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    #test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
    #                                             repeat=False, shuffle=False)

    # Set up a trainer
    #updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    #trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)




if __name__ == '__main__':
    main()
