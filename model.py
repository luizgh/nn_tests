import numpy as np
from layer import *

class FeedForwardNN:
    def __init__(self):
        self.layers = []
    
    def addLayer(self,layer):
        if (len(self.layers) == 0):
            self.nVis = layer.W.shape[0]
            #check for shape?
        self.layers.append(layer)
    
    def fprop(self, X):
        modelState = [X]
        lastLayerActivation = X
        for l in self.layers:
            lastLayerActivation = l.fprop(lastLayerActivation)
            modelState.append(lastLayerActivation)    
        return modelState
    
    def predict(self, X):
        state = self.fprop(X)
        return state[-1]
    
    def calcCost(self, y, modelState, trainParams):
        weightPenalty = float(trainParams["WeightPenalty"])
        hx = modelState[-1]
        m = hx.shape[0]
        
        cost = 1./m * sum(sum(- y * np.log(hx) - (1-y) * np.log(1 - hx)))
        for l in self.layers:
            cost += weightPenalty / (2*m) * l.getWeightCost()
        return cost 
        
    def bprop(self, y, modelState, trainParams):
        weightPenalty = float(trainParams["WeightPenalty"])
        gradients = []
        numLayers = len(self.layers)
        
        hx = modelState[-1]
        m = hx.shape[0]
        
        dLastLayer = hx - y
        
        for layerNum in range(numLayers - 1, 0, -1):
            gradients.insert(0, self.layers[layerNum].getBackPropGrad(modelState[layerNum], dLastLayer, trainParams))
            dLastLayer = self.layers[layerNum].bprop(modelState[layerNum], dLastLayer)
            
        gradients.insert(0, self.layers[0].getBackPropGrad(modelState[0], dLastLayer, trainParams))
        return gradients
    
    def serialize(self):
        return np.concatenate([l.serialize() for l in self.layers])
        
    @staticmethod
    def deserialize(serialized, modelDef):
        m = FeedForwardNN()
        current = 0
        for layer in modelDef:
            l = None
            if layer["name"] == "softmax":
                nInput = layer["nInput"]
                nOutput = layer["nOutput"]
                numberOfVars = nInput*nOutput + nOutput
                l = SigmoidLayer.deserialize(serialized[current: current + numberOfVars], nInput, nOutput)
                current += numberOfVars
            m.layers.append(l)
        return m