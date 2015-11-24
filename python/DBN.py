# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from RBM import RBM
from CRBM import CRBM
from utils import *
import pandas as pd

class DBN(object):
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
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,     # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        #print self.sigmoid_layers[-1].sample_h_given_v().shape
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()



    def pretrain(self, lr=0.1, k=1, epochs=100):
        # pre-train layer-wise
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]
            
            for epoch in xrange(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input,batches=1)
                #cost = rbm.get_reconstruction_cross_entropy()
                #print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost


    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()
        print layer_input
        print numpy.count_nonzero(layer_input)
        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            #print layer_input.shape
            self.log_layer.train(lr=lr, input=layer_input)
            self.finetune_cost = self.log_layer.negative_log_likelihood()
            #print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost
            
            lr *= 0.95
            epoch += 1


    def predict(self, x):
        layer_input = x
        
        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output_continuous(input=layer_input)
        
        out = self.log_layer.predict(layer_input)
        return out



def test_dbn(pretrain_lr=0.1, pretraining_epochs=500, k=1, \
             finetune_lr=0.1, finetune_epochs=20):

    data = pd.read_csv('../../data/alldata.csv',header=None).values
    n_train = int(1.0*len(data))
    X = data[:,:-1]
    X[X>0] = 1
    X[X<=0] = 0
    #X = preprocessing.scale(X)
    #X = (X-numpy.min(X,axis=0))/(numpy.max(X,axis=0)-numpy.min(X,axis=0))
    #print numpy.count_nonzero(X[:,2])
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
    kfold = StratifiedKFold(shuffle=True,y=Y[:n_train],n_folds=10)
    for train_idx,valid_idx in kfold:
        dbn = DBN(input=X[train_idx,:],label=outY[train_idx,:],n_ins=X.shape[1],hidden_layer_sizes=[5],n_outs=outY.shape[1],rng=rng)
        dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
        dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
        thres = sum(Y[train_idx] == 1)*1.0/(sum(Y[train_idx] == -1) + sum(Y[train_idx] == 1))
        print thres
        probs = (dbn.predict(X[train_idx,:]))
        print probs
        y_true = Y[train_idx]
        y_pred = [1 if probs[i][0] > probs[i][1] else -1 for i in xrange(probs.shape[0])]
        print confusion_matrix(y_true, y_pred)
    #dbn = DBN(input=X[trainingidx,:], label=outY[trainingidx,:], n_ins=X.shape[1], hidden_layer_sizes=[10,10], n_outs=outY.shape[1], rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    #dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    #dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
    #print dbn.predict(X[:n_train]).tolist()
    """probs = (dbn.predict(X[n_train:]))
    tp,tn,fp,fn = 0,0,0,0
    for i in xrange(len(probs)):
        if probs[i][0] > probs[i][1]:
            if Y[n_train+i] == 1:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if Y[n_train+i] == -1:
                tn = tn + 1
            else:
                fn = fn + 1
    print tp+fn,tn+fp
    print "Accuracy",(tp+tn)*1.0/(tn+fp+tp+fn)
    print "Up",tp*1.0/(tp+fp),tp*1.0/(tp+fn)
    print "Down",tn*1.0/(tn+fn),tn*1.0/(tn+fp)"""



if __name__ == "__main__":
    test_dbn()