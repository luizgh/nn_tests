import numpy as np
from util import *

def CreateRandomLayer(nUnitsPrevious, nUnitsThisLayer):
    wRange = 1/np.sqrt(nUnitsPrevious)
    W = np.random.uniform(-wRange, wRange, nUnitsPrevious * nUnitsThisLayer).reshape(nUnitsPrevious, nUnitsThisLayer)
    B = np.random.uniform(-wRange, wRange, nUnitsThisLayer)
    return W, B

class Layer:
    pass
    
class SigmoidLayer(Layer):
    @staticmethod
    def CreateRandomLayer( nInput, nOutput, initFunction=CreateRandomLayer):
        layer = SigmoidLayer()
        layer.W, layer.B = initFunction(nInput, nOutput)
        return layer
    
    @staticmethod
    def CreateFromMatrices(W, B):
        layer = SigmoidLayer()
        layer.W = W
        layer.B = B
        return layer
    
    def fprop(self, X):
        return sigmoid(np.dot(X, self.W) + self.B)
        
    def getWeightCost(self):
        return np.sum(self.W * self.W)
    
    def getBackPropGrad(self, layerState, nextLayerGrad, trainParams):
        m = nextLayerGrad.shape[0]
        weightPenalty = float(trainParams["WeightPenalty"])
        
        W_grad = (1./m) * np.dot(layerState.T, nextLayerGrad)
        W_grad += (weightPenalty / m) * self.W
        B_grad = (1./m) * np.sum(nextLayerGrad, axis=0)
        
        return SigmoidLayer.CreateFromMatrices(W_grad, B_grad)
    
    def bprop(self, layerState, nextLayerGrad):
        return layerState * (1 - layerState) * np.dot(nextLayerGrad, self.W.T)
        
    def serialize(self):
        return np.concatenate([self.W.reshape(-1), self.B.reshape(-1)])
    
    @staticmethod
    def deserialize(serialized, nInput, nOutput):
        layer = SigmoidLayer()
        layer.W = serialized[0: nInput*nOutput].reshape(nInput, nOutput)
        layer.B = serialized[nInput*nOutput: nInput*nOutput + nOutput]
        return layer