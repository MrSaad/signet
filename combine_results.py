from __future__ import print_function

from sklearn.metrics import roc_auc_score
from keras.layers import Input, Dense
from keras.models import Model, model_from_json

import argparse
import numpy as np
import os

def train_combined(dataset, c, val_split, batch_size, epochs):
	c = str(c)
	signet = np.loadtxt('results/SigNet/'+dataset+'/C'+c+'/results1.c'+c+'.txt', delimiter=',')
	sprint = np.loadtxt('results/SPRINT/'+dataset+'/C'+c+'/results1.c'+c+'.txt', delimiter=',')
	signet_test = np.loadtxt('results/SigNet/'+dataset+'/C'+c+'/results2.c'+c+'.txt', delimiter=',')
	sprint_test = np.loadtxt('results/SPRINT/'+dataset+'/C'+c+'/results2.c'+c+'.txt', delimiter=',')

	# get y
	train_y = signet[:,1].reshape((signet.shape[0], 1))
	test_y = signet_test[:,1].reshape((signet_test.shape[0], 1))

	# get x
	signet = signet[:,0].reshape((signet.shape[0], 1))
	sprint = sprint[:,0].reshape((sprint.shape[0], 1))
	signet_test = signet_test[:,0].reshape((signet_test.shape[0], 1))
	sprint_test = sprint_test[:,0].reshape((sprint_test.shape[0], 1))

	train_x = np.concatenate((signet, sprint), axis=1)
	test_x = np.concatenate((signet_test, sprint_test), axis=1)

	### MODEL ###
	input = Input(shape=(2,))
	h = Dense(16, activation='relu')(input)
	output = Dense(1, activation='sigmoid')(h)

	model = Model(input, output)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	### TRAIN ###
	model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_split=val_split)

	test_scores = model.predict(test_x, verbose=0)

	print( "SigNet AUC: " + str(roc_auc_score(test_y, signet_test)) )
	print( "SPRINT AUC: " + str(roc_auc_score(test_y, sprint_test)) )
	print( "Combined AUC: " + str(roc_auc_score(test_y, test_scores)) )

	### SAVE ###
	combined_folder = 'results/Combined/'+dataset+'/C'+c+'/'
	model_json = model.to_json()
	with open(combined_folder+'model.json', 'w') as json_file:
	    json_file.write(model_json)
	model.save_weights(combined_folder+'model_weights.h5')
	print('Saved model\n')


def evaluate_combined(dataset, c):
	c = str(c)
	# load model
	combined_folder = 'results/Combined/'+dataset+'/C'+c+'/'
	json_file = open(combined_folder+'model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(combined_folder+'model_weights.h5')
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	for i in range(1, 41):
		print('Combining '+str(i)+' out of 40')
		signet = np.loadtxt('results/SigNet/'+dataset+'/C'+c+'/results'+str(i)+'.c'+c+'.txt', delimiter=',')
		sprint = np.loadtxt('results/SPRINT/'+dataset+'/C'+c+'/results'+str(i)+'.c'+c+'.txt', delimiter=',')

		# get y
		test_y = signet[:,1].reshape((signet.shape[0], 1))

		# get x
		signet = signet[:,0].reshape((signet.shape[0], 1))
		sprint = sprint[:,0].reshape((sprint.shape[0], 1))
		test_x = np.concatenate((signet, sprint), axis=1)

		# evaluate scores
		loss, acc = model.evaluate(test_x, test_y, verbose=0)
		test_scores = model.predict(test_x, verbose=0)
		print( "AUC: " + str(roc_auc_score(test_y, test_scores)) )

		# save to file
		data = np.concatenate((test_scores, test_y), axis=1)
		outfile = combined_folder+'results'+str(i)+'.c'+c+'.txt'
		with open(outfile, 'w') as f:
			np.savetxt(f, data, delimiter=',')


if __name__ == "__main__":

	if not os.path.exists('./results/SigNet/'):
		raise ValueError('SigNet results do not exists, see README on how to download this folder')
	if not os.path.exists('./results/SPRINT/'):
		raise ValueError('SPRINT results do not exists, see README on how to download this folder')

	possible_datasets = ['biogrid', 'hprd', 'innate_manual', 'innate_experimental', 
							'mint', 'int_act', 'park_marcotte']

	parser = argparse.ArgumentParser()
	parser._action_groups.pop()

	# required arguments
	required = parser.add_argument_group('required arguments')
	required.add_argument('-d', '--dataset', type=str, required=True,
		help='string. the dataset to use.', choices=possible_datasets, metavar='')
	required.add_argument('-c', '--ctype', type=int, required=True,
		help='integer. the test type to use. C1, C2 or C3 (enter value between 1-3)', 
		choices=[1,2,3], metavar='')
	
	#optional arguments
	optional = parser.add_argument_group('optional arguments')
	optional.add_argument('-v', '--val_split', type=float, default=0.1,
		help='float. proportion of the training samples to use as ' +
		'cross-validation samples. default=%(default)s', metavar='')
	optional.add_argument('-b', '--batch_size', type=int, default=64,
		help='integer. batch size to use during training. default=%(default)s', metavar='')
	optional.add_argument('-e', '--epochs', type=int, default=50,
		help='integer. the number of epochs to train. default=%(default)s', metavar='')

	args = parser.parse_args()

	if not os.path.exists('./results/Combined/'):
		os.makedirs('./results/Combined/')
	if not os.path.exists('./results/Combined/'+args.dataset):
		os.makedirs('./results/Combined/'+args.dataset)
	if not os.path.exists('./results/Combined/'+args.dataset+'/C'+str(args.ctype)):
		os.makedirs('./results/Combined/'+args.dataset+'/C'+str(args.ctype))


	# Train the combined model
	print('TRAINING Combined Model:')
	train_combined(args.dataset, args.ctype, args.val_split, args.batch_size, args.epochs)

	# Evaluate the combined model on 40 splits
	print('TESTING Combined Model on 40 splits:')
	evaluate_combined(args.dataset, args.ctype)







