import numpy as np
from layer import *
from model import *
from cost import *
from scipy.optimize import fmin_l_bfgs_b
    
nVis=784
nHid=400
nClass=10

layer1 = SigmoidLayer.CreateRandomLayer(nVis, nHid)
layer2 = SigmoidLayer.CreateRandomLayer(nHid, nClass)

model = FeedForwardNN()
model.addLayer(layer1)
model.addLayer(layer2)

trainX = np.load("mnist_train.X.npy")
trainY = np.load("mnist_train.Y.npy")
testX = np.load("mnist_test.X.npy").reshape(-1, nVis)
testY = np.load("mnist_test.Y.npy")

X = trainX.reshape(60000, -1)
y = np.zeros((trainY.shape[0], nClass))
for i in range(trainY.shape[0]):
    y[i, trainY[i]] = 1

trainParams = {"WeightPenalty" : 10}
modelDef = [
{'name' : 'softmax', 'nInput' : 784, 'nOutput' : 400 },
{'name' : 'softmax', 'nInput' : 400, 'nOutput' : 10 }]

params = model.serialize()
    
calcCost = FeedForwardNNCostFunction(X, y, modelDef, trainParams)

res = fmin_l_bfgs_b(func = calcCost.getCost, approx_grad=False, x0 = params, maxiter = 100, disp = True)

best_params = res[0]
best_model = FeedForwardNN.deserialize(best_params, modelDef)

probs = best_model.predict(testX)
preds = probs.argmax(axis=1)

accuracy = sum(preds == testY) / (preds.shape[0] * 1.0)

print "Test set accuracy: %.2f" % accuracy
