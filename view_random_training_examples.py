import numpy as np
import csv
from matplotlib import pyplot as plt
import random

# Load all training inputs to a python list
train_inputs = []
with open('train_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
        for pixel in train_input[1:]: # Start at index 1 to skip the Id
            train_input_no_id.append(float(pixel))
        train_inputs.append(train_input_no_id) 

# Load all training ouputs to a python list
train_outputs = []
with open('train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        train_output_no_id = int(train_output[1])
        train_outputs.append(train_output_no_id)

# Keep displaying random examples until stopped 
while True:
    rand_idx = random.randint(0,len(train_inputs)-1)
    print "Index: %i, Output: %i" % (rand_idx, train_outputs[rand_idx])
    # Convert to numpy array and display as image
    example_input = np.asarray(train_inputs[rand_idx])
    reshaped_input = np.reshape(example_input, (48,48))
    plt.imshow(reshaped_input, cmap="Greys_r")
    plt.show()