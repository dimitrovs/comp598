import os

import csv
import numpy as np

from itertools import izip


# This class provides a data management API
class Data(object):
	'''
		To use this class, make sure that the following raw data files
		have been placed in /data/raw:

			- train_input.csv
			- train_output.csv
			- test_input.csv

		This class provides a data loading and transforming service.
		It reads a raw form of the images dataset, and returns a 
		more convenient structured form.  The format of the data is always
		the same, but there are various pre-processing and weighting 
		options.  The data format is a list of 'examples', and each example
		is a tuple, having the format (`id`, `feature values`, `class_name`).
		So it looks something like this:

			[
				...
				('id', [x_1, x_2, ...], 'class'),
				...
			]

		You can get output like that by calling any of the methods that
		match 'as_*'.

		All the methods also accept a `data_part` keyword argument, which 
		can take the values 'test' or 'train' (it is 'train' by default).
		This determines whether the testing or training data will be returned.
	'''


	# Constants
	NUM_CLASSES = 10
	RAW_DIR = 'raw'
	RAW_MERGED = os.path.join(RAW_DIR, 'train_merged.csv')
	RAW_INPUT = os.path.join(RAW_DIR, 'train_inputs.csv')
	RAW_OUTPUT = os.path.join(RAW_DIR, 'train_outputs.csv')
	RAW_TEST = os.path.join(RAW_DIR, 'test_inputs.csv')


	def __init__(self, data_dir='../data', limit=None, verbose=True):
		self.data_dir = data_dir
		self.verbose = verbose
		self.limit = limit

		# Merges the training examples with their classes and outputs to file.
		def merge_input_output():
			input_reader = csv.reader(
				open(os.path.join(data_dir, Data.RAW_INPUT), 'r'))
			output_reader = csv.reader(
				open(os.path.join(data_dir, Data.RAW_OUTPUT), 'r'))
			
			writer = csv.writer(
				open(os.path.join(data_dir, Data.RAW_MERGED), 'w'))

			# Write a new file that merges the info from both
			for input_row, output_row in izip(input_reader, output_reader):
				writer.writerow(input_row + output_row[1:])


		# Check if the merged data file exists, if not, make it
		if not os.path.isfile(os.path.join(data_dir, self.RAW_MERGED)):
			self.say('merging training input and output on the first '
				'use of Data...')
			merge_input_output()

		# Load raw data into memory
		reader_train = csv.reader(
			open(os.path.join(data_dir, self.RAW_MERGED), 'r'))
		reader_test = csv.reader(
			open(os.path.join(data_dir, self.RAW_TEST), 'r'))

		# The files have headers, so advance both readers by one line
		reader_train.next()
		reader_test.next()

		# Converts the feature values in a row from the reader
		def to_vect(row):
			# First entry is the id
			vect = [row[0]]
			for pixel in row[1:-1]:
				# vect.append(np.float64(pixel))
				vect.append(float(pixel))
			# Append last entry, which is the class
			vect.append(row[-1])
			return vect

		# limit can be used to limit the amount of data loaded
		if limit is not None:
			self.data = map(to_vect,
				[reader_train.next() for i in range(limit)])

			# Get the number of test examples
			with open(os.path.join(data_dir, self.RAW_TEST), 'r') as f:
				num_test_examples = len(f.readlines()) - 1

			# If the number of test examples is less than the limit, then
			# include all test examples
			if num_test_examples < limit:
				self.test_data = map(to_vect, [row for row in reader_test])
			else:
				self.test_data = map(to_vect,
					[reader_test.next() for i in range(limit)])

		else:
			self.data = map(to_vect, [row for row in reader_train])
			self.test_data = map(to_vect, [row for row in reader_test])


	def say(self, string):
		if self.verbose:
			print string


	def as_grayscale_vals(self, data_part='train'):
		'''
		As raw grayscale values, with not pre-processing.
		'''

		# Use the test set or training set, depending on what was requested
		if data_part == 'train':
			data_set = self.data
		elif data_part == 'test':
			data_set = self.test_data
		else:
			raise ValueError("data_part must be either 'train' or 'test'")

		self.say("outputting as grayscale values...")

		return_data = []
		for row in data_set:

			if data_part == 'train':
				idx, feature_vect, class_name = row[0], row[1:-1], row[-1]
				
				return_data.append((idx, feature_vect, class_name))
			else:
				idx, feature_vect = row[0], row[1:]

				return_data.append((idx, feature_vect))

		return return_data

