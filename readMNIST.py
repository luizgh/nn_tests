import struct 
import numpy

def readMNISTFile(filename):
    f = open(filename, "rb")
    f.seek(4)
    m = struct.unpack('>i', f.read(4))[0]
    nrow = struct.unpack('>i', f.read(4))[0]
    ncol = struct.unpack('>i', f.read(4))[0]

    examples = numpy.zeros((m, 28,28))

    for iExample in range(m):
        thisExample = f.read(ncol * nrow)
        thisExampleBytes = [struct.unpack('B', thisByte)[0] for thisByte in thisExample]
        examples[iExample,:,:] = numpy.asarray(thisExampleBytes).reshape(28,28)
    
    return examples


def readMNISTFileLabels(filename):
    f = open(filename, "rb")
    f.seek(4)
    m = struct.unpack('>i', f.read(4))[0]
    data = f.read(m)
    labelBytes = [struct.unpack('B', thisByte)[0] for thisByte in data]
    labels = numpy.asarray(labelBytes)
    return labels

    
examples = readMNISTFile("train-images-idx3-ubyte")
labels = readMNISTFileLabels("train-labels-idx1-ubyte")
  
numpy.save("mnist_train.X", examples)
numpy.save("mnist_train.Y", labels)

examples = readMNISTFile("t10k-images-idx3-ubyte")
labels = readMNISTFileLabels("t10k-labels-idx1-ubyte")
  
numpy.save("mnist_test.X", examples)
numpy.save("mnist_test.Y", labels)



trainX = numpy.load("mnist_train.X.npy")
trainY = numpy.load("mnist_train.Y.npy")

X = trainX.reshape(60000, -1)
y = trainY
