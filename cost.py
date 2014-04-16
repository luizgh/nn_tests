from model import FeedForwardNN

class FeedForwardNNCostFunction:
    def __init__ (self, X, y, modelDef, trainParams):
        self.X = X
        self.y = y
        self.modelDef = modelDef
        self.trainParams = trainParams
        
    def getCost (self, rolledParams):
        model = FeedForwardNN.deserialize(rolledParams, self.modelDef)
        
        state = model.fprop(self.X)
        cost = model.calcCost(self.y, state, self.trainParams)
        grads = model.bprop(self.y, state, self.trainParams)
        gradModel = FeedForwardNN()
        gradModel.layers = grads
        
        return cost, gradModel.serialize()