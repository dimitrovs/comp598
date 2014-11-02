import numpy as np
import csv
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

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
#test_inputs = []
#with open('test_inputs.csv', 'rb') as csvfile:
#    reader = csv.reader(csvfile, delimiter=',')
#    next(reader, None)  # skip the header
#    for test_input in reader:
#        test_input_no_id = []
#        for pixel in test_input[1:]: # Start at index 1 to skip the Id
#            test_input_no_id.append(float(pixel))
#        test_inputs.append(test_input_no_id)


clf = GaussianNB()
#clf.fit(train_inputs, train_outputs)
#test_outputs = clf.predict(test_inputs)

# Write a random output for every test_input
#test_output_file = open('test_output_nb.csv', "wb")
#writer = csv.writer(test_output_file, delimiter=',')
#writer.writerow(['Id', 'Prediction']) # write header
#for idx, test_output in enumerate(test_outputs):
#    row = [idx+1, test_output]
#    writer.writerow(row)
#test_output_file.close()

print "Shape: {0}".format(len(train_inputs[0]))
scores = cross_validation.cross_val_score(clf, train_inputs, train_outputs, cv=10,n_jobs=1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

components = 100
while components < 2304:
	clf = GaussianNB()
	pca = PCA(n_components=components)
	train_inputs_new = pca.fit_transform(train_inputs,train_outputs)
	print "Shape: {0}".format(len(train_inputs_new[0]))
	scores = cross_validation.cross_val_score(clf, train_inputs_new, train_outputs, cv=10,n_jobs=1)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	components += 50

