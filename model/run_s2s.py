from __future__ import print_function
from collections import defaultdict
import operator
import numpy as np
import chainer
from chainer import cuda
import math
import time
import chainer.links as L
import sys
import chainer.functions as F
from Trainer import Trainer
from Create_Model import Create_Model
import optparse
import argparse
import glob
from chainer import training, cuda,optimizer, optimizers, serializers,Variable
from chainer.training import extensions
from sklearn.cross_validation import train_test_split
import datetime
import random
import re
import os
import time


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
parser.add_argument('--att_type', type=str,default='dot', help='[in/out] model file')
parser.add_argument("--lr", default=0.001, help='Initial learning rate',type=float)
parser.add_argument("--lr-descend", default=1.8, help='Learning rate descending',type=float)
parser.add_argument("--lr-stop", default=0.00001, help='Learning rate stop',type=float)
parser.add_argument("--depth", dest="depth", default=1,
        type=int, help="Number of LSTM layers in both encoder and decoder")
parser.add_argument("--drop", dest="drop", default=0,
        type=float, help="Drop ratio")
parser.add_argument("--grad-clip", dest="grad_clip", default=5,
        type=int)
parser.add_argument('--bprob-len', type=int, default=35, help='length of truncated BPTT')
parser.add_argument('--out-model', help='output model directory')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

max_len=800

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))

print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))


#separate test and train


#import pdb; pdb.set_trace()


src_files = glob.glob('./corpus/es/*')
tgt_files = glob.glob('./corpus/en/*')
src_files.sort()
tgt_files.sort()
data_size=len(src_files)
data=np.array([src_files,tgt_files])
data=data.transpose()
data=data[random.sample(xrange(len(data)), data_size)]

dev=random.sample(xrange(len(data)), args.batchsize)
dev_data=data[dev]
data=np.delete(data,dev,0)
test=random.sample(xrange(len(data)), args.batchsize)
test_data=data[test]
data=np.delete(data,test,0)

data=data.transpose()
dev_data=dev_data.transpose()
test_data=test_data.transpose()

print(len(data))
x =np.genfromtxt(src_files[0], delimiter=" ").astype(np.float32)
y =np.genfromtxt(tgt_files[0], delimiter=" ").astype(np.float32)

input_units=len(x[0])
output_units=len(y[0])

#init trainer

trainer = Trainer(grad_clip=args.grad_clip, opt_type="Adam", lr=args.lr, lr_descend=args.lr_descend, lr_thres=args.lr_stop)
nnet = Create_Model(input_units, args.hidden_units, output_units,args.depth,args.drop, att_type=args.att_type)
trainer.set_model(nnet)
if(args.gpu >= 0):
    cuda.check_cuda_available()
    if(args.gpu >= 0) :
        xp = cuda.cupy
        nnet.to_gpu()
        print ("use GPU")
    else :
        np
        print("use cpu")

def create_batch(source,target, batchsize):
    src_batch=[]
    tgt_batch=[]
    file_list = []

    batch_num=0
#    import pdb; pdb.set_trace()
    for src_file, tgt_file in zip(source,target):

        x =np.genfromtxt(src_file, delimiter=" ").astype(xp.float32)
        y =np.genfromtxt(tgt_file, delimiter=" ").astype(xp.float32)
        src=xp.zeros((max_len+1,x.shape[1])).astype(xp.float32)
        tgt=xp.zeros((max_len+1,y.shape[1])).astype(xp.float32)

#        import pdb; pdb.set_trace()
        #print(test.shape)
        if(len(x)<max_len and len(y)<max_len):
            for i in range(len(x)):
                src[i+1] =cuda.to_cpu(x[i]).all()
            for k in range(len(y)):
                tgt[k+1]=cuda.to_cpu(x[i]).all()

#            import pdb; pdb.set_trace()
            #if(len(x)>max_len):
            #    max_len=len(x)
            #    src_batch.insert(0,src)
            #    tgt_batch.insert(0,tgt)
            #else:
            src_batch.append(src)
            tgt_batch.append(tgt)
            file_list.append(tgt_file)
            batch_num+=1
            if(batch_num>=batchsize):
                #import pdb; pdb.set_trace()
                #print("yield batch")
                yield src_batch,tgt_batch,file_list
                src_batch=[]
                tgt_batch=[]
                batch_num=0
#        print(batch_num)
    #print("yield batch EOF")
    #import pdb; pdb.set_trace()
    yield src_batch,tgt_batch,file_list

def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()

def train(args):

  output=[]
  #batch_list=create_batch(src_files,tgt_files,args.batchsize);
  epoch=0
  time_flag=True

  #import pdb; pdb.set_trace()
  while epoch < args.epoch:
    #trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
    trained = 0
    batch_num=0
    total_loss=0
    #data.random()
    for src_batch, tgt_batch ,_ in create_batch(data[0],data[1],args.batchsize):
     # import pdb; pdb.set_trace()
      if(time_flag):
         start = time.time()
      src_batch=Variable(xp.array(np.array(src_batch).astype(xp.float32)))
      tgt_batch=Variable(xp.array(np.array(tgt_batch).astype(xp.float32)))

      src_batch=F.transpose(src_batch,(1,0,2))
      tgt_batch=F.transpose(tgt_batch,(1,0,2))

      nnet.encode(src_batch)
      sen_loss = 0

      for t in tgt_batch:

          o = nnet.decode(t)

          loss_t = F.mean_squared_error(o, t)
          # loss_i = loss_func(o, t)
          sen_loss += loss_t.data
          #if train:
          trainer.add_loss(loss_t)

          #if collect_result:
          #import pdb; pdb.set_trace()


          trainer.update_params()

      #if collect_result:
      #import pdb; pdb.set_trace()

      #output_mask.append(y_mask)
      current_loss=(sen_loss) / tgt_batch.shape[0]
      total_loss += (sen_loss) / tgt_batch.shape[0]
      trainer.update_params()

      if(time_flag):
          elapsed_time = time.time() - start
          t_time=elapsed_time*int(len(data[0])/args.batchsize)*args.epoch
          hour=int(t_time/(60**2))
          minutes=int((t_time%60**2)/60)
          #print("it takes %d hour %d minute",hour,minutes
          print ("it takes %s hours %s minutes" % (hour, minutes))
          time_flag=False
      print ("epoch batch_num & loss",epoch,batch_num,current_loss)
      #if train:
      batch_num+=1
      break
    total_loss /= len(data[0])/args.batchsize
    print("epoch dev_loss",test(dev_data,nnet,mode="dev"))

    print("epoch test_loss", test(test_data,nnet,epoch=epoch))

def test(data,model,mode="test",epoch=0):
    output=[]
    res_dir="result/en/"+str(epoch)
    if(mode!="dev"):
        if(not os.path.exists(res_dir)):
            os.mkdir(res_dir)

    for src_batch, tgt_batch, file_name in create_batch(data[0],data[1],args.batchsize):

        src_batch=Variable(xp.array(np.array(src_batch).astype(xp.float32)))
        tgt_batch=Variable(xp.array(np.array(tgt_batch).astype(xp.float32)))
	import pdb; pdb.set_trace()
        src_batch=F.transpose(src_batch,(1,0,2))
        tgt_batch=F.transpose(tgt_batch,(1,0,2))

        nnet.encode(src_batch)

     #t = chainer.Variable(xp.zeros(tgt_batch.shape[1], dtype=np.int32), volatile='auto')
        sen_output = []
        sen_loss = 0



        for t in tgt_batch:

            o = nnet.decode(t)

            loss_t = F.mean_squared_error(o, t)
            if mode != "dev" :
                sen_output.append(o.data)

         # loss_i = loss_func(o, t)
            sen_loss += loss_t.data


        if(mode != "dev"):
            sen_output=Variable(np.array(sen_output))
            sen_output=F.transpose(sen_output,(1,0,2))

            for f in range(len(file_name)):

                mgc=file_name[f].replace("corpus_debug/en",res_dir)
                #np.save(mgc,output[f])
                np.savetxt(mgc, sen_output.data[f])
                #print(mgc)

    total_loss = (sen_loss) / len(tgt_batch)


    return total_loss


if __name__ == '__main__':
    train(args)
