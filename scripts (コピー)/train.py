from __future__ import print_function
import numpy as np
import chainer
from chainer import cuda
import time
import chainer.links as L
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../model')
import chainer.functions as F
from Trainer import Trainer
from Create_Model import Create_Model
import glob
from chainer import training, cuda,optimizer, optimizers, serializers,Variable
from chainer.training import extensions
import re
import os
import configparser

#settings
args = sys.argv
inconf=configparser.SafeConfigParser()
inconf.read(args[1])

data_dir        = inconf.get('settings','numpy_dir')
train_src_list  = glob.glob(data_dir+'train/src/*')
train_trg_list  = glob.glob(data_dir+'train/trg/*')
dev_src_list  = glob.glob(data_dir+'dev/src/*')
dev_trg_batchs  = glob.glob(data_dir+'dev/trg/*')
test_src_list  = glob.glob(data_dir+'test/src/*')
test_trg_batchs  = glob.glob(data_dir+'test/trg/*')
src_max_len     = int(inconf.get('settings',"src_max_len"))
trg_max_len     = int(inconf.get('settings',"trg_max_len"))
src_dim         = int(inconf.get('settings',"src_dim"))
trg_dim         = int(inconf.get('settings',"trg_dim"))
batchsize       = int(inconf.get('trainer','batchsize'))
epoch           = int(inconf.get('trainer','epoch'))
gpu             = int(inconf.get('gpu','gpu'))
out_dir         = inconf.get('trainer','out_dir')
units           = int(inconf.get('trainer','units'))
hidden_units    = int(inconf.get('trainer','hidden_units'))
model           = inconf.get('trainer','model')
att_type        = inconf.get('trainer','att_type')
lr              = float(inconf.get('trainer','learning_ratio'))
lr_descend      = float(inconf.get('trainer','lr_descend'))
lr_stop         = float(inconf.get('trainer','lr_stop'))
depth           = int(inconf.get('trainer','depth'))
grad_clip       = int(inconf.get('trainer','grad_clip'))
bprob_len       = int(inconf.get('trainer','bprob_len'))
drop            = inconf.get('trainer','drop')
log             = inconf.get('trainer','log_dir')

train_src_list.sort()
train_trg_list.sort()
dev_src_list.sort()
dev_trg_batchs.sort()
test_src_list.sort()
test_trg_batchs.sort()


#prepare
print("prepare done")
trainer = Trainer(grad_clip=grad_clip, opt_type="Adam", lr=lr, lr_descend=lr_descend, lr_thres=lr_stop)
nnet = Create_Model(src_dim, hidden_units, trg_dim,depth,drop, att_type=att_type)
trainer.set_model(nnet)
if(gpu >= 0):
    cuda.check_cuda_available()
    if(gpu >= 0) :
        xp = cuda.cupy
        nnet.to_gpu()
        print ("use GPU",gpu)
    else :
        np
        print("use cpu")
#learning method

def batch_gen(src_files,trg_file):
    x=xp.zeros((batchsize,src_max_len,src_dim))
    y=xp.zeros((batchsize,trg_max_len,trg_dim))
    batch_num=0
    for src,trg in zip(src_files, trg_file):
        x_tmp=xp.array(xp.load(src).astype(xp.float32))
        y_tmp=xp.array(xp.load(src).astype(xp.float32))
        if x_tmp.shape[0]>=src_max_len or y_tmp.shape[0]>=trg_max_len:
            continue
        x[batch_num,0:x_tmp.shape[0]]=x_tmp

        y[batch_num,0:y_tmp.shape[0]]=y_tmp
        batch_num+=1
        if batch_num==batchsize:
            x=F.transpose(x,(1,0,2))
            y=F.transpose(y,(1,0,2))
            batch_num=0
            yield Variable(x),Variable(y)
    x=F.transpose(x,(1,0,2))
    y=F.transpose(y,(1,0,2))
    return Variable(x),Variable(y)

def learn(src_files,trg_file,mode="train"):
    total_loss=0
    for x,y in batch_gen(src_files,trg_file):
        import pdb; pdb.set_trace()

        nnet.encode(x)
        sen_loss = 0

        for t in y:
            o = nnet.decode(t)
            gomi,a = t.data.nonzero()
            a=a+1
            if len(gomi)<o.data.shape[0]:
                j=xp.zeros(o.data.shape[0])
                for i,g in enumerate(gomi):
                    j[g]=a[i]
                a=Variable(j.astype(xp.int32))
            else:
                a=Variable(a.astype(xp.int32))


            loss_t = F.softmax_cross_entropy(o, a)
            sen_loss += loss_t.data

            if mode is "train":
                trainer.add_loss(loss_t)
                trainer.update_params()
        total_loss += (sen_loss)
        #print(total_loss/len(trg_file))
    return total_loss/len(trg_file)


#train
e=0
champ=sys.float_info.max
while e < epoch:

    if e ==0:
        start = time.time()
    print(e)
    loss=learn(train_src_list,train_trg_list,"train")
    if e ==0:
        total=time.time()-strt
        hour=int(total/(60**2))
        minutes=int((total%60**2)/60)
        print("it will cost ",hour)
    print("train loss:",loss)
    loss_dev=learn(dev_src_list,dev_trg_batchs,"dev")
    print("dev loss:",loss_dev)
    if loss_dev < champ:
        champ=loss_dev
        prefix=log_dir+"champ"
        serializers.save_spec(prefix,nnet)

   #trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
    e+=1
