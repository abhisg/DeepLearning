import rbm
import pandas as pd
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

data = pd.read_csv('../../data/alldata.csv',header=None).values
n_train = int(0.9*len(data))
#normalise the data for RBM
eps = 0.001
X = data[:,:-1]
X[X>0] = 1
X[X<=0] = 0
print X.shape
Y = data[:,-1].astype(int)
outY = numpy.zeros((X.shape[0],2))
for i in xrange(Y.shape[0]):
	if Y[i] == 1:
		outY[i][0] = 1
	else:
		outY[i][1] = 1
outY = numpy.array(outY)
dbn = rbm.DBN(X.shape[1],[50,30],[0,0,0],batches=100)
dbn.train(X[:n_train],500)
model = LogisticRegression(max_iter=20)
C = [10,20,30,40,50,100,200]
for c in C:
	#model = SVC(C=c)
	model = LogisticRegression(max_iter=c)
	model.fit(dbn.run_visible(numpy.array(X[:n_train].tolist())),Y[:n_train])
	probs =  model.predict(dbn.run_visible(numpy.array(X[n_train:].tolist())))
	print confusion_matrix(probs,Y[n_train:])
