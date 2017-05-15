##############################################################################
# SIGNET
# Saad Ahmed
#
# A Siamese Convolutional Neural Network Architecture for Predicting 
# Protein-Protein Interactions
##############################################################################

from __future__ import print_function

from sklearn.metrics import roc_auc_score
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.callbacks import Callback
from itertools import izip, count

import argparse
import pickle
import numpy as np
import sys
import os

np.random.seed(1337)  # For reproducability 

def run_signet(dataset, split, val_split=0.05, batch_size=128, epochs=5):
	dataset_folder = './data/'+dataset+'/'
	if dataset == 'park_marcotte':
		signature_file = './data/signatures_parkmarcotte.pickle'
	else:
		signature_file = './data/signatures.pickle'

	##############################################################################
	# PREPROCESS DATA 
	##############################################################################

	# Build signature vectors for each protein
	print("Loading signatures for proteins")
	with open(signature_file, 'rb') as handle:
		save_data = pickle.load(handle)
		prot_sigs = save_data["sigs"]
		sig_len = save_data["sig_len"]

	def create_set(filename, val_split=None, shuffle=True):

		num_samples = sum(1 for line in open(filename))
		x1 = np.zeros((num_samples, sig_len, 1), dtype=np.float32)
		x2 = np.zeros((num_samples, sig_len, 1), dtype=np.float32)
		y = np.zeros((num_samples, 1), dtype=np.float32)

		with open(filename) as f:
			for i ,line in izip(count(),f):
				label, prot1, prot2 = line.strip().split('\t')
				x1[i] = prot_sigs[prot1]
				x2[i] = prot_sigs[prot2]
				y[i][0] = 1 if label == 'Positive' else 0

		if shuffle:
			print("Shuffling samples")
			permutation = np.random.permutation(y.shape[0])
			x1 = x1[permutation]
			x2 = x2[permutation]
			y = y[permutation]

		if val_split is not None: 
			cv_break = int((1 - val_split) * num_samples)
			return x1[:cv_break], x2[:cv_break], y[:cv_break], x1[cv_break:], x2[cv_break:], y[cv_break:]
		else:
			return x1, x2, y


	print('Generating training and cross-validation sets')
	train_x1, train_x2, train_y, valid_x1, valid_x2, valid_y = create_set(
	    dataset_folder+'train.'+str(split)+'.txt', 
	    val_split=val_split, shuffle=True)

	print('Generating test sets')
	test_c1_x1, test_c1_x2, test_c1_y = create_set(
	    dataset_folder+'test.c1.'+str(split)+'.txt', shuffle=False)
	test_c2_x1, test_c2_x2, test_c2_y = create_set(
	    dataset_folder+'test.c2.'+str(split)+'.txt', shuffle=False)
	test_c3_x1, test_c3_x2, test_c3_y = create_set(
	    dataset_folder+'test.c3.'+str(split)+'.txt', shuffle=False)


	##############################################################################
	# DEFINE MODEL 
	##############################################################################
	print("Creating model")

	# define single protein model
	prot_input = Input(shape=(sig_len, 1))
	out = Conv1D(64, 3, padding='same')(prot_input)
	out = Conv1D(64, 3, padding='same')(prot_input)
	out = MaxPooling1D(pool_size=3)(out)
	out = Dropout(0.3)(out)
	out = Conv1D(128, 3, padding='same')(out)
	out = MaxPooling1D(pool_size=3)(out)
	out = Dropout(0.2)(out)
	out = Conv1D(256, 3, padding='same')(out)
	out = MaxPooling1D(pool_size=3)(out)
	out = Dropout(0.2)(out)
	out = Flatten()(out)
	out = Dense(1024, activation='relu')(out)

	prot_model = Model(inputs=prot_input, outputs=out)

	# create a siamese model by concatenating two single-protein models
	prot_a = Input(shape=(sig_len, 1))
	prot_b = Input(shape=(sig_len, 1))
	out_a = prot_model(prot_a)
	out_b = prot_model(prot_b)
	concatenated = Concatenate()([out_a, out_b])

	# final predictions
	out = Dense(1024, activation='relu')(concatenated)
	out = Dropout(0.1)(out)
	out = Dense(1024, activation='relu')(out)
	preds = Dense(1, activation='sigmoid')(out)

	# compiling the model
	model = Model(inputs=[prot_a, prot_b], outputs=preds)
	model.compile(optimizer='adam', loss='binary_crossentropy', 
					metrics=['accuracy'])


	##############################################################################
	# TRAIN AND TEST MODEL 
	##############################################################################

	### TRAIN ###
	if (valid_x1.shape[0] == 0):
		model.fit([train_x1, train_x2], train_y, batch_size=batch_size, 
			epochs=epochs, verbose=1)
	else:
		model.fit([train_x1, train_x2], train_y, batch_size=batch_size, 
			epochs=epochs, verbose=1, validation_data=([valid_x1, valid_x2], valid_y))

	### TEST ###
	def printTestStats(stat_type, test_x1, test_x2, test_y):
		print(stat_type+"  RESULTS:")

		# Accuracy
		score, acc = model.evaluate([test_x1,test_x2], test_y, 
									batch_size=batch_size, verbose=0)
		print("\ttest accuracy: "+str(acc))

		# AUC
		test_scores = model.predict([test_x1,test_x2], 
									batch_size=batch_size, verbose=0)
		print( "\tAUC: " + str(roc_auc_score(test_y, test_scores)) )

		# write to file
		cfile = stat_type.upper()
		data = np.concatenate((test_scores, test_y.astype(np.int32)), axis=1)
		outfile = './results/SigNet/'+dataset+'/'+cfile+'/results'+str(split) + '.'+stat_type+'.txt'
		with open(outfile, 'w') as f:
			np.savetxt(f, data, delimiter=',')

	printTestStats('c1', test_c1_x1, test_c1_x2, test_c1_y)
	printTestStats('c2', test_c2_x1, test_c2_x2, test_c2_y)
	printTestStats('c3', test_c3_x1, test_c3_x2, test_c3_y)


if __name__ == '__main__':
	possible_datasets = ['biogrid', 'hprd', 'innate_manual', 'innate_experimental', 
							'mint', 'int_act', 'park_marcotte']

	parser = argparse.ArgumentParser(description='signet.py')
	parser._action_groups.pop()

	# required arguments
	required = parser.add_argument_group('required arguments')
	required.add_argument('-d', '--dataset', type=str, required=True,
		help='string. the dataset to use.', choices=possible_datasets, metavar='')
	required.add_argument('-s', '--split', type=int, required=True,
		help='integer. the dataset split to use. value between 1 and 40.',
		choices=range(1,41), metavar='')
	
	#optional arguments
	optional = parser.add_argument_group('optional arguments')
	optional.add_argument('-v', '--val_split', type=float, default=0.05,
		help='float. proportion of the training samples to use as ' +
		'cross-validation samples. default=%(default)s', metavar='')
	optional.add_argument('-b', '--batch_size', type=int, default=128,
		help='integer. batch size to use during training. default=%(default)s', metavar='')
	optional.add_argument('-e', '--epochs', type=int, default=5,
		help='integer. the number of epochs to train. default=%(default)s', metavar='')

	args = parser.parse_args()

	# make appropriate directories
	if not os.path.exists('./results/SigNet/'):
		os.makedirs('./results/SigNet/')
	if not os.path.exists('./results/SigNet/'+args.dataset):
		os.makedirs('./results/SigNet/'+args.dataset)
		os.makedirs('./results/SigNet/'+args.dataset+'/C1')
		os.makedirs('./results/SigNet/'+args.dataset+'/C2')
		os.makedirs('./results/SigNet/'+args.dataset+'/C3')

	# Run Signet
	run_signet(args.dataset, args.split, val_split=args.val_split, 
				batch_size=args.batch_size, epochs=args.epochs)




