import numpy as np
from layer import *
from model import *

def checkGradients(model, trainParams, nIterations = 1000, verbose = False):
    epsilon = 1e-4
    sampleX = np.random.rand(100 * 200).reshape(100,200)
    sampleY = np.zeros((100,10))
    sampleY[:,0] = 1
       
    for i in xrange(nIterations):
        selectedLayerNumber = np.random.randint(0, len(model.layers))
        selectedLayer = model.layers[selectedLayerNumber]
        layerParams = vars(selectedLayer)
        availableParams = layerParams.keys()
        
        selectedParam = availableParams[np.random.randint(0, len(availableParams))]
        
        maxR = layerParams[selectedParam].shape[0]
        r = np.random.randint(0, maxR)
        item = r
        if (len(layerParams[selectedParam].shape) >1):
            maxC = layerParams[selectedParam].shape[1]
            c = np.random.randint(0, maxC)
            item = (r,c)
        
        orig = layerParams[selectedParam][item]
        
        if(verbose):
            print "Checking layer %d, param %s, item %s" % (selectedLayerNumber, selectedParam, item)
        
        grads = model.bprop(sampleY, model.fprop(sampleX), trainParams)
        
        layerParams[selectedParam][item] = orig + epsilon
        
        costPlus = model.calcCost(sampleY, model.fprop(sampleX), trainParams)
        
        layerParams[selectedParam][item] = orig - epsilon
        
        costMinus = model.calcCost(sampleY, model.fprop(sampleX), trainParams)
        
        layerParams[selectedParam][item] = orig
        
        numericalGrad = (costPlus - costMinus)/(2*epsilon)
        analiticalGrad =  vars(grads[selectedLayerNumber])[selectedParam][item]
        
        if (abs(numericalGrad - analiticalGrad) > 1e-04):
            print "Gradient checking failed. Layer %d, param %s, item %s" % (selectedLayerNumber, selectedParam, item)
            assert False
            
    print "Gradients OK"

if __name__ == "__main__":            
    layer1 = SigmoidLayer.CreateRandomLayer(200,100)
    layer2 = SigmoidLayer.CreateRandomLayer(100,50)
    layer3 = SigmoidLayer.CreateRandomLayer(50,60)
    layer4 = SigmoidLayer.CreateRandomLayer(60,10)

    model = FeedForwardNN()
    model.addLayer(layer1)
    model.addLayer(layer2)
    model.addLayer(layer3)
    model.addLayer(layer4)
    
    trainParams = {"WeightPenalty" : 10}
    checkGradients(model, trainParams, 10000, verbose=True)
