from __future__ import print_function

import argparse

from itertools import product
import pickle
import numpy as np
import sys
import os
from tqdm import tqdm 

# the types of amino acids found in the park_marcotte dataset
pm_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
		'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# the types of amino acids found in all other datasets
other_aas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
			'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']


def prot_to_sig(prot, data_type, kmer_length=3, symmetric=True):
	"""Converts a protein to a signature vector, as defined by Martin et. Al

	# Arguments
		prot: list. The protein to be converted
		kmer_length: integer. Length of each kmer to consider for
					signature computation. must be positive and odd
		symmetric: boolean. Whether or not ordering of neighbours
					for each signature value matters
	# Returns
		numpy array. The signature representation of the protein
		integer. The number of signatures computed for this protein

	# Raises
		ValueError: In case kmer_length is not positive and odd
	"""

	# determine of parkmarcotte dataset is being used or not
	if data_type == 'pm':
		aas = pm_aas
	else:
		aas = other_aas

	if kmer_length < 1 or kmer_length % 2 == 0:
		raise ValueError('kmer_length must be positive and odd')

	# create dictionary of possible kmers
	if symmetric:
		neighbours = sorted(list(set([''.join(sorted(p)) for p in product(aas,
			repeat=kmer_length - 1)])))
		kmers=[aa + n  for aa in aas for n in neighbours]
		kmer_dict = dict(zip(kmers, range(len(kmers))))
	else:
		kmers = [''.join(p) for p in product(aas, repeat=kmer_length)]
		kmer_dict = dict(zip(kmers, range(len(kmers))))

	sig_len = len(kmer_dict)

	# map protein to signature vector using kmer dictionary
	sig = np.zeros((sig_len,), dtype=np.float32)
	for i in range(len(prot) - (kmer_length - 1)):
		kmer = prot[i:i + kmer_length]
		if symmetric:
			kmer = kmer[len(kmer) / 2] + ''.join(sorted(kmer[:len(kmer) / 2] +
				kmer[len(kmer) / 2 + 1:]))
		sig[kmer_dict[kmer]] += 1

	return sig.reshape((sig_len, 1)), sig_len


def compute_sigs(infile, outfile, data_type, kmer_length=3, symmetric=True):
	"""Computes protein signatures for a given set of proteins
	# Arguments
		infile: input filename for proteome
		outfile: output filename for saved signatures
		data_type: pm (parkmarcotte) or other
		kmer_length: Length of each kmer to consider for
					signature computation. must be positive and odd
		symmetric: Whether or not ordering of neighbours
					for each signature value matters
	"""
	proteins = dict()
	prot_sigs = dict()

	num_lines = sum(1 for line in open(infile, 'r'))

	with open(infile, 'r') as f:
		for line in tqdm(f, total=num_lines/2, ncols=75, bar_format=
			'{percentage:3.0f}%|{bar}|Protein {n_fmt}/{total_fmt}'):

			# get protein name and sequence
			prot_name = line[1:].strip()
			prot_seq = f.next().strip()

			# store sequence in protein dictionary
			proteins[prot_name] = prot_seq

			# store signature in signature dictionary
			prot_sigs[prot_name], sig_len = prot_to_sig(prot_seq,
				data_type, kmer_length=kmer_length, symmetric=symmetric)

	save_data = dict()
	save_data["sigs"] = prot_sigs
	save_data["sig_len"] = sig_len

	print("Saving Signatures to file")
	with open(outfile, 'wb') as handle:
		pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(kmer_length, symmetric):

	# compute signatures for park_marcotte dataset
	print('Computing signatures for park_marcotte dataset')
	compute_sigs('data/human_seq_parkmarcotte.fasta', 'data/signatures_parkmarcotte.pickle', 
		'pm', kmer_length=kmer_length, symmetric=symmetric)

	# compute signatures for all other datasets
	print('Computing signatures for all other datasets')
	compute_sigs('data/human_seq.fasta', 'data/signatures.pickle', 
		'other', kmer_length=kmer_length, symmetric=symmetric)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-k', '--kmer_length', type=int, default=3,
		help='integer. length of each kmer to consider for signature computation. ' +
		'must be positive and odd. default=%(default)i', metavar='')
	parser.add_argument('-s', '--symmetric', type=bool, default=True,
		help='bool. define whether ordering of neighbours matters. ' +
		'default=%(default)s', metavar='')
	args = parser.parse_args()

	main(args.kmer_length, args.symmetric)


