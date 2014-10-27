# Add this project's src folder to the path
import os
import sys
sys.path.append(os.path.abspath('../'))

import math
import numpy as np
import random
import time

from collections import defaultdict

import data_processing as dp


def calc_gaussian(u, sig_sq, x):
	'''
	Calculate Gaussian function (In normal distribution) at x, with mean u and
	variance sig_sq.
	'''
	mult_term = 1 / (2 * math.pi * sig_sq) ** 0.5
	exp_term = math.exp((-1) * ((x - u) ** 2) / (2 * sig_sq))
	return mult_term * exp_term


class GaussianNaiveBayesClassifier(object):
	'''
	
	Gaussian Naive Bayes Classifier.

	Training consists of providing the classifier a set of examples having
	the structure:

		('id', [x_1, x_2, ...], 'class_name')

	The order and number of features is expected to be constant throughout all
	examples.
	'''

	def __init__(self):
		# Feature vectors by class
		self.fvects_by_class = defaultdict(list)

		# Mean and variance values across features for each set of examples
		# with same class
		self.means_by_class = {}
		self.variances_by_class = {}

		# Prior probabilities of each class
		self.priors = {}
	

	def add_example(self, example):
		'''
		Add an example for training.
		'''

		feature_vect, class_name = example[1], example[2]
		
		self.fvects_by_class[class_name].append(feature_vect)


	def get_num_examples(self):
		'''
		Return number of training examples.
		'''

		num_examples = 0
		for _, class_feature_vs in self.fvects_by_class.iteritems():
			num_examples += len(class_feature_vs)
		return num_examples


	def get_num_features(self):
		'''
		Return the number of features.
		'''

		for _, class_feature_vs in self.fvects_by_class.iteritems():
			return len(class_feature_vs[0])


	def train(self, examples):
		'''
		Train classifier.
		'''

		# Add examples for training
		for example in examples:
			self.add_example(example)

		num_examples = self.get_num_examples()
		num_features = self.get_num_features()

		for class_name, class_feature_vs in self.fvects_by_class.iteritems():
			
			# Calculate prior
			self.priors[class_name] = (float(len(class_feature_vs)) /
				num_examples)

			# Calculate mean vector
			mean_v = []
			for i in range(num_features):
				mean_v.append(np.mean([ v[i] for v in class_feature_vs ]))
			self.means_by_class[class_name] = mean_v

			# Calculate variance vector
			variance_v = []
			for i in range(num_features):
				variance_v.append(np.var([ v[i] for v in class_feature_vs ]))
			self.variances_by_class[class_name] = variance_v


	def classify(self, feature_vect):
		'''
		Takes in an example_feature vector (which is missing last component,
		the class name), and outputs the likelihood maximizing class, based on
		the assumption that features are conditionally independent of one
		another.
		'''

		num_features = self.get_num_features()

		class_scores = {}

		# For each class, calculate a score equal to the likelihood that the
		# class would produce this feature vector
		for class_name, mean_v in self.means_by_class.iteritems():
			variance_v = self.variances_by_class[class_name]

			class_scores[class_name] = np.float64(self.priors[class_name])

			for i in range(num_features):
				(u, sig_sq, x) = mean_v[i], variance_v[i], feature_vect[i]
				# Conditional probability of x given the class under the
				# Gaussian distribution
				cond_prob = calc_gaussian(u, sig_sq, x)

				class_scores[class_name] *= np.float64(cond_prob)

		# Predicted class is the class yielding the maximum likelhiood
		predicted_class = sorted(class_scores.items(), key=lambda t: -t[1])[0][0]
		return predicted_class


class CrossValTester(object):
	'''
	Given a data set, allows one to perform cross validation with the 
	GaussianNaiveBayesClassifier.

	The dataset must be a list of examples, where each example has the form

		(<id>, <feature_vector>, <class_name>)

	id should be a string, feature_counts should be a list with numeric values,
	and class_name should be a string.

	So the dataset should look something like this:

		[
			...
			('1', [x_1, x_2, ...], '5'),
			...
		]
	'''


	def __init__ (self, dataset, limit=None):
		self.dataset = dataset

		# Randomize the examples' ordering.
		random.shuffle(self.dataset)

		# Optionally limit data for improved time efficiency
		if limit is not None:
			self.dataset = self.dataset[:limit]

		self.size = len(self.dataset)


	def extract_train_and_test_set (self, fold, test_set_size, is_last=True):
		'''
		Partitions the examples into a training and test set.
		'''

		# Select the index range of examples to be used as the test set
		startIdx = fold * test_set_size
		endIdx = startIdx + test_set_size
		if is_last:
			endIdx = None

		test_set = self.dataset[startIdx:endIdx]

		# Select the remaining examples for the training set
		train_set = self.dataset[:startIdx] + self.dataset[endIdx:]

		return (train_set, test_set)


	def cross_validate(self, k=None):
		'''
		Divide the dataset set into k equal folds (If k doesn't divide the
		number of examples evenly, then the folds won't all be equal). For
		each 'fold' of cross validation, train a GaussianNaiveBayesClassifier
		on all the data outside the fold, then test it on the data inside the
		fold, and repeat for all folds.  Keep a running tally of the number
		of classifications correct.
		'''

		# If k is not specified, do leave-one-out cross validation
		if k is None:
			k = self.size
		k = int(k)

		test_set_size = self.size / k

		self.score = 0

		for fold in range(k):

			print 'fold %i' % (fold + 1)

			is_last = (fold is k - 1)
			(train_set, test_set) = self.extract_train_and_test_set(fold,
				test_set_size, is_last)

			# Create and train GaussianNaiveBayesClassifier.
			classifier = GaussianNaiveBayesClassifier()
			classifier.train(train_set)

			# Apply classifier on the test set and tally score
			for example in test_set:
				prediction_class = classifier.classify(example[1])
				if prediction_class == example[2]:
					self.score += 1

		accuracy = self.score / float(self.size)

		# print 'OVERALL ACCURACY: %f' % accuracy

		# Return the overall accuracy 
		return accuracy


class CrossValCase(object):
	'''
	Runner for cross-validation.
	'''

	# Permissable representations of the dataset.
	ALLOWED_REPRESENTATIONS = ['as_grayscale_vals']

	# Number of folds.
	K = 10


	def __init__(self):
		pass


	def run(
		self,
		representation,
		limit=None
	):
		'''
		Generate dataset and apply cross-validation.
		'''

		assert(representation in self.ALLOWED_REPRESENTATIONS)
		data_manager = dp.Data(limit=limit)

		# This requests the data manager to calculate the desired
		# representation of the data
		dataset = getattr(data_manager, representation)(data_part='train')

		cross_val_tester = CrossValTester(dataset=dataset)
		accuracy = cross_val_tester.cross_validate(self.K)

		# Return the accuracy of the cross-validation
		return accuracy


if __name__ == '__main__':
	# Number of training examples to consider is the first command-line
	# argument, if specified
	limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

	if limit is None:
		print 'Cross-validation'
	else:
		print 'Cross-validation on %d examples' % limit

	start_time = time.time()

	accuracy = CrossValCase().run(
		representation='as_grayscale_vals',
		limit=limit
	)

	print 'Accuracy: %0.3f' % accuracy


	print 'Time elapse: %0.2f ms', time.time() - start_time
