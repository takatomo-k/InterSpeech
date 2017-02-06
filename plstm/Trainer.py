from __future__ import print_function
import sys
import chainer
import component as com
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

class Trainer(object):
    def __init__(self, grad_clip=5., opt_type="Adam", lr=0.001, lr_descend=1.8, lr_thres=0.00001):
        print("~~~~~ Training settings ~~~~~~~")
        print('~ Inital learning rate: %5f' % (lr))
        print('~ learning rate descend rate: %f' % (lr_descend))
        print('~ threshold of stop training: %5f' % (lr_thres))

        if opt_type == 'Adam':
            print('~ Optimizer: Adam')
            self.optimizer = optimizers.Adam()
        elif opt_type == 'RMSpropGraves':
            print('~ Optimizer RMSpropGraves')
            self.optimizer = optimizers.RMSpropGraves(lr=lr)
        else:
            print('Optimizer %s not supported' % opt_type)
            exit(1)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        self.grad_clip = grad_clip
        self.lr_thres = lr_thres
        self.accum_loss = 0
        self.lr_descend = lr_descend
        self.model = None

    def continue_train(self):
        if self.optimizer.lr < self.lr_thres:
            return False
        else:
            return True

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        self.optimizer.lr = lr

    def descend_lr(self):
        print('Descending')
        self.optimizer.lr /= self.lr_descend

    def set_model(self, model):
        self.model = model
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(self.grad_clip))

    def update_params(self):
        if not isinstance(self.accum_loss, int):
            self.model.zerograds()
            self.accum_loss.backward()
            #self.accum_loss.unchain_backward()  # truncate
            self.accum_loss = 0
            self.optimizer.update()

    def add_loss(self, loss):
        self.accum_loss += loss

    def reset_with_update():
         self.model.reset()
         self.accum_loss.backward()
         self.accum_loss.unchain_backward()
         self.optimizer.update()
