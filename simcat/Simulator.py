#!/usr/bin/env python

"""
Pulls a slice from the database and runs simulation to get SNP counts.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

import h5py
import toytree
import numpy as np
from scipy.special import comb
from phymsim import Model
from ast import literal_eval


#############################################################################
class Simulator:
    """
    This is the object that runs on the engines by loading data from the HDF5,
    building the msprime simulations calls, and then calling .run() to fill
    count matrices and return them.
    """
    def __init__(self, database_file, slice0, slice1, mutator='msprime', run=True):

        # location of data
        self.database = database_file
        self.slice0 = slice0
        self.slice1 = slice1

        # parameter transformations
        self.mut = 1e-8

        self.mutator = mutator

        # open view to the data
        with h5py.File(self.database, 'r') as io5:

            # sliced data arrays
            self.node_heights = io5["node_heights"][slice0:slice1, ...]
            self.node_Nes = io5["node_Nes"][slice0:slice1, ...]
            self.admixture_args = io5["admixture_args"][slice0:slice1, ...]

            # attribute metadata
            self.tree = toytree.tree(io5.attrs["tree"])
            self.nsnps = io5.attrs["nsnps"]
            self.ntips = len(self.tree)
            #self.aedges = self.asources.shape[1]

            # storage for output
            self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb
            self.nvalues = self.slice1 - self.slice0
            self.counts = np.zeros(
                (self.nvalues, self.nquarts*16*16), dtype=np.int64) 

        # calls run and returns filled counts matrix
        if run:
            self.run()

    def _return_Model(self, idx):
        # get tree
        tree = self.tree
        # assign times to nodes

        # assign Nes to nodes
        for node in tree.treenode.traverse():
            node.add_feature('Ne', self.node_Nes[idx, node.idx])

        mut = self.mut
        # interpret argument:
        ad_arg = literal_eval(self.admixture_args[idx].decode())
        # define model
        return(Model(tree,
                     Ne=None,
                     mut=mut,
                     recomb=0,
                     admixture_edges=ad_arg))

    def run(self):
        for idx in range(self.nvalues):
            sim = self._return_Model(idx)
            self.counts[idx] = sim._run_snps(self.nsnps)
