import numpy as np
import numpy 
import math
import gc
from scipy.optimize import minimize


def CreateRandomLayer(nUnitsPrevious, nUnitsThisLayer):
    wRange = 1/math.sqrt(nUnitsPrevious)
    W = np.random.uniform(-wRange, wRange, nUnitsPrevious * nUnitsThisLayer).reshape(nUnitsPrevious, nUnitsThisLayer)
    B = np.random.uniform(-wRange, wRange, nUnitsThisLayer)
    return W, B

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def CalculateCostFunction(X, y, model, weightPenalty):
    # X.shape == (nExamples, nVis)
    # Y.shape == (nExamples)
    
    #model.W1 == (nVis, nH1)
    #model.B1 == (nH1)
    #model.W2 == (nH1, nH2)
    #model.B2 == (nH2)
    
    #Perform forward prop
    m, n  = X.shape
    weightPenalty = float(weightPenalty)
    
    a1 = X
    z2 = np.dot(a1, model['W1']) + model['B1']
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, model['W2']) + model['B2']
    a3 = sigmoid(z3)
    
    hx = a3
    cost = 1./m * sum(sum(- y * np.log(hx) - (1-y) * np.log(1 - hx)))
    cost += (weightPenalty / (2*m) ) * (np.sum(model['W1'] * model['W1']) + np.sum(model['W2'] * model['W2']))
    
    d3 = hx - y
    d2 = a2 * (1 - a2) * np.dot(d3, model["W2"].T)
    
    W2_grad = (1./m) * np.dot(a2.T, d3)
    W2_grad += (weightPenalty / m) * model['W2']
    B2_grad = (1./m) * np.sum(d3, axis=0)
    W1_grad = (1./m) * np.dot(a1.T, d2)
    W1_grad += (weightPenalty / m) * model['W1']
    B1_grad = (1./m) * np.sum(d2, axis=0)
    
    grads = {'W1' : W1_grad, 'W2' : W2_grad, 'B1' : B1_grad, 'B2' : B2_grad};
    return cost, grads


def predict(X, model):
    # X.shape == (nExamples, nVis)
    # Y.shape == (nExamples)
    
    #model.W1 == (nVis, nH1)
    #model.B1 == (nH1)
    #model.W2 == (nH1, nH2)
    #model.B2 == (nH2)
    
    #Perform forward prop
    m, n  = X.shape
    
    a1 = X
    z2 = np.dot(a1, model['W1']) + model['B1']
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, model['W2']) + model['B2']
    a3 = sigmoid(z3)
    
    return a3

def checkGradients(costFunction, verbose = False, weightPenalty = 0.01):
    rngState = numpy.random.get_state()
    W1, B1 = CreateRandomLayer(200,100)
    W2, B2 = CreateRandomLayer(100, 10)
    epsilon = 1e-4
    
    sampleModel = {'W1': W1, 'B1' : B1, 'W2' : W2, 'B2' : B2}
    sampleX = numpy.random.rand(100 * 200).reshape(100,200)
    sampleY = numpy.zeros((100,10))
    sampleY[:,0] = 1
    
    availableParams = sampleModel.keys()
    for i in xrange(1000):        
        param = availableParams[numpy.random.randint(0, len(availableParams))]
        maxR = sampleModel[param].shape[0]
        r = numpy.random.randint(0, maxR)
        item = r
        if (len(sampleModel[param].shape) >1):
            maxC = sampleModel[param].shape[1]
            c = numpy.random.randint(0, maxC)
            item = (r,c)
        orig = sampleModel[param][item]
        
        if(verbose):
            print "Checking %s, item %s" % (param, item)
        
        cost, grad = costFunction(sampleX, sampleY, sampleModel,weightPenalty)
        
        sampleModel[param][item] = orig + epsilon
        costPlus, gradPlus = costFunction(sampleX, sampleY, sampleModel, weightPenalty)
        
        sampleModel[param][item] = orig - epsilon
        costMinus, gradMinus = costFunction(sampleX, sampleY, sampleModel, weightPenalty)
        
        sampleModel[param][item] = orig
        
        numericalGrad = (costPlus - costMinus)/(2*epsilon)
        analiticalGrad = grad[param][item]
        
        if (abs(numericalGrad - analiticalGrad) > 1e-04):
            print "Gradient checking failed. Num: %f. Analit: %f" % (numericalGrad, analiticalGrad)
            print "Param: %s, item: %s, State: %s" % (param, item, rngState)
            assert False
    print "Gradients OK"

class CalculateCost:
    def __init__ (self, X, y, weightPenalty):
        self.X = X
        self.y = y
        self.weightPenalty = weightPenalty
        
    def getCost (self, rolledParams):
	model = UnrollParams(rolledParams)
        cost, grads = CalculateCostFunction(self.X, self.y, model, self.weightPenalty)
        return cost, RollParams(grads)
        
    
    
def RollParams(model):
    return np.concatenate([model['W1'].reshape(-1), 
                           model['B1'].reshape(-1), 
                           model['W2'].reshape(-1), 
                           model['B2'].reshape(-1)])
    
def UnrollParams(rolledParams):
    model = {}
    start = 0
    model['W1'] = rolledParams[start : start + nVis * nHid].reshape(nVis, nHid)
    start += nVis * nHid
    model['B1'] = rolledParams[start : start + nHid]
    start += nHid
    model['W2'] = rolledParams[start : start + nHid * nClass].reshape(nHid, nClass)
    start += nHid * nClass
    model['B2'] = rolledParams[start : start + nClass]
    return model
    
nVis=784
nHid=400
nClass=10

model = dict()
model['W1'], model['B1'] = CreateRandomLayer(nVis,nHid)
model['W2'], model['B2'] = CreateRandomLayer(nHid, nClass)
variables = RollParams(model)

trainX = np.load("mnist_train.X.npy")
trainY = np.load("mnist_train.Y.npy")

X = trainX.reshape(60000, -1)
y = np.zeros((trainY.shape[0], nClass))
for i in range(trainY.shape[0]):
    y[i, trainY[i]] = 1

    
checkGradients(CalculateCostFunction, True)

calcCost = CalculateCost(X, y, 1.)

#res = minimize(fun = calcCost.getCost, x0 = variables , method="L-BGFS-B", jac=True, options = { "maxiter" : 400, "disp" : True})
res = fmin_l_bfgs_b(func = calcCost.getCost, approx_grad=False, x0 = variables, maxiter = 100, disp = True)


best_model = UnrollParams(res[0])

preds = predict(X,best_model)
preds = preds.argmax(axis=1)

sum(preds == testy) / (preds.shape[0] * 1.0)

def loadImg(filename):
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = 255 - img
    imgRes = cv2.resize(img, (28,28))
    return imgRes.reshape(-1)

y = numpy.asarray([8,4,0,6,7,1,9,2])
numbers = numpy.asarray( [loadImg("%d.png" % f) for f in y])
preds = predict(numbers, best_model).argmax(axis=1)
acc = sum(preds == y) / (y.shape[0]*1.0)

