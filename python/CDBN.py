# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from RBM import RBM
from CRBM import CRBM
from DBN import DBN
from utils import *
import pandas as pd

 
class CDBN(DBN):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 rng=None):
        
        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if rng is None:
            rng = numpy.random.RandomState(1234)

        
        assert self.n_layers > 0


        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        rng=rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)

            # construct rbm_layer
            if i == 0:
                rbm_layer = CRBM(input=layer_input,     # continuous-valued inputs
                                 n_visible=input_size,
                                 n_hidden=hidden_layer_sizes[i],
                                 W=sigmoid_layer.W,     # W, b are shared
                                 hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layer_sizes[i],
                                W=sigmoid_layer.W,     # W, b are shared
                                hbias=sigmoid_layer.b)
                
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()



def test_cdbn(pretrain_lr=0.1, pretraining_epochs=100, k=1, \
             finetune_lr=0.1, finetune_epochs=50):

    data = pd.read_csv('../data/alldata.csv',header=None).values
    n_train = int(0.9*len(data))
    X = data[:,:-1]
    Y = data[:,-1].astype(int)
    outY = numpy.zeros((X.shape[0],2))
    for i in xrange(Y.shape[0]):
        if Y[i] == 1:
            outY[i][0] = 1
        else:
            outY[i][1] = 1
    outY = numpy.array(outY)
    #print outY.tolist()
    #print X.shape,Y.shape
    
    rng = numpy.random.RandomState(123)
    #print X[n_train:]
    # construct DBN
    dbn = CDBN(input=X[:n_train,:], label=outY[:n_train,:], n_ins=X.shape[1], hidden_layer_sizes=[20,10], n_outs=outY.shape[1], rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
    #print dbn.predict(X[:n_train])
    probs = (dbn.predict(X[n_train:]))
    print dbn.predict(X[:n_train,:])
    #print probs
    tp,tn,fp,fn = 0,0,0,0
    for i in xrange(len(probs)):
        if probs[i][0] > probs[i][1]:
            if Y[i] == 1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if Y[i] == -1:
                tn = tn + 1
            else:
                fp = fp + 1
    print tp+fn,tn+fp
    print "Accuracy",(tp+tn)*1.0/(tn+fp+tp+fn)
    print "Up",tp*1.0/(tp+fp),tp*1.0/(tp+fn)
    print "Down",tn*1.0/(tn+fn),tn*1.0/(tn+fp)




if __name__ == "__main__":
    test_cdbn()
