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
from tqdm import tqdm
import gc
#settings
args = sys.argv
inconf=configparser.SafeConfigParser()
inconf.read(args[1])

data_dir        = inconf.get('settings','numpy_dir')
src_dir         = data_dir+"/train/src/"
trg_dir         = data_dir+"/train/trg/"
basename="TTSEN"

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
data_num=0
#prepare
print("prepare done")
#trainer = Trainer(grad_clip=grad_clip, opt_type="Adam", lr=lr, lr_descend=lr_descend, lr_thres=lr_stop)
nnet = Create_Model(src_dim, hidden_units, trg_dim,depth,drop, att_type=att_type)
#trainer.set_model(nnet)
if(gpu >= 0):
    cuda.check_cuda_available()
    if(gpu >= 0) :
        xp = cuda.cupy
        cuda.get_device(gpu).use()
        nnet.to_gpu()
        print ("use GPU",gpu)
    else :
        np
        print("use cpu")
#learning method

def batch_gen(mode):
    if mode is "train":
        start=0
        stop =50000
    elif mode is "dev":
        start=50000
        stop =55000
    elif mode is "test":
        start=55500
        stop =60000

    x_batch=xp.zeros((batchsize,src_max_len,src_dim)).astype(xp.float32)
    y_batch=xp.zeros((batchsize,trg_max_len,trg_dim)).astype(xp.float32)

    #import pdb; pdb.set_trace()
    batch_num=0
    data_num=0
    for index in range(start,stop):
        #i=np.random.randint(start,stop)
        i=index
        src=src_dir+basename+str(i)+".npy"
        trg=trg_dir+basename+str(i)+".npy"
        x_tmp=xp.load(src).astype(xp.float32)
        y_tmp=xp.load(trg).astype(xp.float32)



        if x_tmp.shape[0]+1>src_max_len or y_tmp.shape[0]+1>trg_max_len:
            #print(x_tmp.shape[0],y_tmp.shape[0])
            continue
        #print(i)
        x_batch[batch_num,1:x_tmp.shape[0]+1]=x_tmp
        x_batch[batch_num,0]=xp.ones(src_dim).astype(xp.int32)
        for trg_id in range(trg_max_len):
            if trg_id < y_tmp.shape[0]-1:
                y_batch[batch_num,trg_id]=y_tmp[trg_id]
                #a=np.nonzero(y_tmp[trg_id])
            else:
                y_batch[batch_num,trg_id]=y_tmp[-2]
                #a=np.nonzero(y_tmp[-2])
            #print(a[0],trg_id,y_tmp.shape[0])
        batch_num+=1
        if batch_num==batchsize:
            data_num+=batch_num
            batch_num=0
            x=F.transpose(x_batch,(1,0,2))
            y=F.transpose(y_batch,(1,0,2))
            x_batch=xp.zeros((batchsize,src_max_len,src_dim)).astype(xp.float32)
            y_batch=xp.zeros((batchsize,trg_max_len,trg_dim)).astype(xp.float32)
            #import pdb; pdb.set_trace()
            #a,b=xp.nonzero(y.data)


            yield x,y

    #x=F.transpose(x_batch,(1,0,2))
    #y=F.transpose(y_batch,(1,0,2))
    #data_num+=batch_num
    #print("data_num is",data_num)
    #yield x,y

def learn(model,mode="train"):
    total_loss=0
    pbar=tqdm(batch_gen(mode))
    batch_num=0
    model.reset()
    data_num=0
    for x,y in pbar:

        model.encode(x)
        sen_loss = 0
        sen_id=0
        for t in y:
            o = model.decode(t)

            ids,value=t.data.nonzero()
            if len(ids)<batchsize:
                import pdb; pdb.set_trace()

            t=Variable(xp.array(value).astype(xp.int32))

            #import pdb; pdb.set_trace()
            loss_t = F.softmax_cross_entropy(o, t)

            #print(o.data[0].argmax(0),t.data[0],sen_id)
            sen_id+=1


            if loss_t.data<100:
                sen_loss += loss_t.data
                data_num +=1
            else:
                print("num output",loss_t.data)
                #print(loss_t.data)
            if mode is "train" and loss_t.data<100:
                trainer.add_loss(loss_t)
        if mode is "train" and loss_t.data<100:
            trainer.update_params()

        model.reset()
        total_loss += (sen_loss)
        batch_num+=1
        pbar.set_description("Loss %s" % ((sen_loss)))
        #print(total_loss/len(trg_file))




    return total_loss/data_num,model


#test

e=0
champ=sys.float_info.max
while e < epoch:

    prefix=log+str(e)

    if os.path.exists(prefix):
        serializers.load_hdf5(prefix, nnet)
        print("epoch",e)
        loss,nnet=learn(nnet,"dev")
        print("train loss:",loss)


    #print("start dev")
    #loss,nnet=learn(nnet,"dev")
    #print("dev loss:",loss)
    #if loss_dev < champ:
    #    champ=loss_dev
    #prefix=log+str(e)
    #serializers.save_hdf5(prefix,nnet)

   #trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
    e+=1
