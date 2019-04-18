#!/usr/bin/env python

"""
Pulls a slice from the database and runs simulation to get SNP counts.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

import h5py
import os
import toytree
import numpy as np
import msprime as ms
import itertools as itt
from scipy.special import comb
from _msprime import LibraryError
from keras.models import load_model

from .jitted import count_matrix_int, mutate_jc
from .Model import Model
from .Simulator import Simulator
from .parallel import Parallel
from .utils import get_all_admix_edges, SimcatError, Progress, tile_reps
from .Database import Database
from .ml import Analysis

#############################################################################

class Streamer:
	""" 
	This is the object that runs on the engines by loading data from the HDF5,
	building the msprime simulations calls, and then calling .run() to fill
	count matrices and return them. 
	"""
	def __init__(self,
		name,
		num_dbs,
	    workdir,
	    starting_tree,
	    model,
	    nsnps=10000,
	    nedges=0,
	    theta=0.01,
	    seed=123,
	    admix_prop_min=0.05,
	    admix_prop_max=0.50,
	    admix_edge_min=0.5, 
	    admix_edge_max=0.5,
	    exclude_sisters=False,
	    force=False,
	    quiet=False,):

		# location of databases and model
		self.workdir = (
        workdir if workdir 
        else os.path.realpath(os.path.join('.', "databases")))
		self.name = name
		self.model = model
		self.model_path = os.path.realpath(os.path.join(self.workdir,self.name+'_mod.h5'))
		self.model.save(self.model_path)

		self.nsnps = nsnps
		self.nedges = nedges
		self.theta = theta
		self.seed = seed
		self.admix_prop_min = admix_prop_min
		self.admix_prop_max = admix_prop_max
		self.admix_edge_min = admix_edge_min
		self.admix_edge_max = admix_edge_max
		self.exclude_sisters = exclude_sisters
		self.force = force
		self.quiet = quiet

		self.starting_tree = starting_tree

	def generate_dat(self):
		while 1:
			db_idx=str(np.random.randint(100000000))
			tree = self.starting_tree
			# or tree slider
			dbname = self.name + '_tmpdb_' + str(db_idx)
			tmpdb = Database(name=dbname,
				workdir = self.workdir,
				tree = tree,
				admix_edge_min=self.admix_edge_min,
				admix_edge_max=self.admix_edge_max,
				admix_prop_min=self.admix_prop_min,
				admix_prop_max=self.admix_prop_max,
				nedges=self.nedges,
				ntests=1,
				nreps=1,
				nsnps=1000,
				theta=self.theta,
				seed=self.seed,
				force=self.force,
				quiet=True
				)
			tmpdb.run()
			data = Analysis(name=dbname, workdir=self.workdir, scale=1, run=False,quiet=True)

			if os.path.isfile(data.db_counts):
			    os.remove(data.db_counts)
			else:    ## Show an error ##
			    print("Error: %s file not found" % data.db_counts)

			if os.path.isfile(data.db_labels):
			    os.remove(data.db_labels)
			else:    ## Show an error ##
			    print("Error: %s file not found" % data.db_labels)

			yield (data.X,data.y)


	def run(self):
		if self.model:
			# let's make sure we're loading from file, to continue
			del self.model
		training_model = load_model(self.model_path)
		for db_idx in range(self.num_dbs):
			tree = starting_tree
			# or tree slider
			dbname = self.name + 'tmpdb' + str(db_idx)
			tmpdb = Database(name=dbname,
				workdir = self.workdir,
				tree = tree
				)
			tmpdb.run()


















