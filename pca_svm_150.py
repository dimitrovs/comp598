import numpy as np
import csv
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm

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
        train_output_no_id =  int(train_output[1])
        train_outputs.append(train_output_no_id)

# Load all test inputs to a python list
test_inputs = []
with open('test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for test_input in reader:
        test_input_no_id = []
        for pixel in test_input[1:]: # Start at index 1 to skip the Id
            test_input_no_id.append(float(pixel))
        test_inputs.append(test_input_no_id)

clf = svm.SVC()
pca = PCA(n_components=150)
train_inputs_new = pca.fit_transform(train_inputs,train_outputs)
clf.fit(train_inputs_new, train_outputs)
test_inputs_new = pca.transform(test_inputs)
test_outputs = clf.predict(test_inputs_new)

# Write a random output for every test_input
test_output_file = open('test_output_pca_svm_150.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',')
writer.writerow(['Id', 'Prediction']) # write header
for idx, test_output in enumerate(test_outputs):
    row = [idx+1, test_output]
    writer.writerow(row)
test_output_file.close()

