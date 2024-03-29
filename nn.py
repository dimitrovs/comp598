import csv
import numpy as np
import math
import random
from sklearn import cross_validation

"neural network parameters"
#number of cells in hidden layer
hid_layer = 70
#size of training_set and test_size (totalling at most 50 000)
train_size = 50000
test_size = 0
"Stochastic gradient parameters"
#number of epochs, size of SGD batch, and eta
epoch = 30
batch = 10
eta = 0.1

def get_data(file1, file2, train_num = 50000, test_num = 0):
    """ get_data takes the file containing the training inputs and the
        training outputs and returns two lists.
        
        The input list contains, for each example, a numpy array (2034x1)
        containing the 2304 pixels of the image in float
        
        The output list contains, for each example,
        the integer (0-9) it represents.
        """
    train_inputs = []
    test_inputs = []
    with open(file1, 'rb') as f:
        reader =csv.reader(f, delimiter = ',')
        next(reader) #skip header row
        index = 0
        for train_input in reader:
            pixels = []
            for pixel in train_input[1:]: #Start at index 1 to skip Id number
                pixels.append([float(pixel)])
            if index < train_num:
                train_inputs.append(np.array(pixels))
            elif index < train_num + test_num:
                test_inputs.append(np.array(pixels))
            else:
                break
            index += 1

    train_outputs = []
    test_outputs = []
    with open(file2, 'rb') as f:
        reader = csv.reader(f, delimiter = ',')
        next(reader)
        index = 0 
        for train_output in reader:
            if index < train_num:
                train_outputs.append(int(train_output[1]))
            elif index < train_num + test_num:
                test_outputs.append(int(train_output[1]))
            else:
                break
            index += 1

    return train_inputs, train_outputs, test_inputs, test_outputs

class Network:

    def __init__(self, size_of_layers):
        self.num_layers = len(size_of_layers)
        self.biases = [np.random.randn(y,1) for y in size_of_layers[1:]]
        self.weights = [np.random.randn(x,y) for x, y in \
                        zip(size_of_layers[1:], size_of_layers[:-1])]


    def feedforward(self, train_input):
        """The train_input is a single 2034x1 numpy array containing the pixels
            of this particular input example.
            The function then outputs the z values and activation values
            for each cell of each layer in the network."""
        #activations will contain all activations of the network given the input
        activations = [train_input]
        #zs will contain the z values of the network given the input
        zs = []
        #the train_input set is just a list of the pixels
        #We transform it to a 2034x1 numpy array for matrix multiplication operations
        activation = train_input
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i],activation) + self.biases[i]
            zs.append(z)
            activation = vector_sigmoid(z)
            activations.append(activation)
        return zs, activations

    def SGD(self, train_inputs, train_outputs, epochs, batch_size, eta,
            test_input = None, test_output = None):
        train_set = zip(train_inputs, train_outputs)
        n = len(train_set)
        k = batch_size
        for j in range(epochs):
            random.shuffle(train_set)
            for i in range(n/k):
                batch = train_set[i*k: (i+1)*k]
                gradient_w = [np.zeros(np.shape(W)) for W in self.weights]
                gradient_b = [np.zeros(np.shape(B)) for B in self.biases]
                for x,y in batch:
                    delta_w, delta_b = self.backprop(x,y)
                    gradient_w = [a+b for a,b in zip(gradient_w, delta_w)]
                    gradient_b = [a+b for a,b in zip(gradient_b, delta_b)]
                self.weights = [a - (eta/k) * b for a,b in zip(self.weights, gradient_w)]
                self.biases = [a - (eta/k) * b for a,b in zip(self.biases, gradient_b)]
            if test_input:
                print "Epoch {}: {} / {}".format(
                    j, self.test(test_input, test_output), len(test_input))
            else:
                print "epoch {} done.".format(j+1)        


    def backprop(self, pixels, result):
        zs, activations = self.feedforward(pixels)
        delta_w = [np.zeros(np.shape(W)) for W in self.weights]
        delta_b = [np.zeros(np.shape(B)) for B in self.biases]
        change = activations[-1]
        change[result - 1, 0] -= 1
        # '*' is the hadamard product when multiplying numpy arrays
        delta = change * vector_sigmoid_prime(zs[-1])
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta
        for i in range(2, self.num_layers):
            z = zs[-i]
            delta = self.weights[-i+1].transpose().dot(delta) * \
                    vector_sigmoid_prime(z)
            delta_w[-i] = np.dot(delta, activations[-i-1].transpose())
            delta_b[-i] = delta
        return delta_w,delta_b
            
            
    def test(self, test_input, test_output):
        total = 0
        for x,y in zip(test_input, test_output):
            for w, b in zip(self.weights, self.biases):
                x = vector_sigmoid(np.dot(w, x) + b)
            if y == np.argmax(x):
                total += 1
        return total
                
                    
                
            
            
            
        
        


def sigmoid(z):
    return (1/(1+math.exp(-z)))

def vector_sigmoid(vector):
    #this function is used to apply the above sigmoid function to numpy arrays
    func = np.vectorize(sigmoid)
    return func(vector)

def sigmoid_prime(z):
    return (sigmoid(z)*(1-sigmoid(z)))

def vector_sigmoid_prime(vector):
    #this function is used to apply the above sigmoid_prime function to numpy arrays
    func = np.vectorize(sigmoid_prime)
    return func(vector)
                
        
        
        



######## MAIN #########

train_inputs, train_outputs, test_inputs, test_outputs = \
              get_data("./data_and_scripts/train_inputs.csv", \
                       "./data_and_scripts/train_outputs.csv", 50000)

X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
    train_inputs, train_outputs, test_size=0.2, random_state=0)

net = Network([2304,hid_layer,10])

net.SGD(X_train, y_train, epoch, batch, eta, X_test, y_test)






#### get_data, feedforward, backprop and constructor are working
#### and have been tested
