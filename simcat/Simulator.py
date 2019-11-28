#!/usr/bin/env python

"""
Pulls a slice from the database and runs simulation to get SNP counts.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

import h5py
import itertools

import ipcoal
import toytree
import numpy as np
from .utils import get_snps_count_matrix



class IPCoalWrapper:
    """
    This is the object that runs on the engines by loading data from the HDF5,
    building the msprime simulations calls, and then calling .run() to fill
    count matrices and return them.
    """
    def __init__(self, database_file, slice0, slice1, run=True):

        # location of data
        self.database = database_file
        self.slice0 = slice0
        self.slice1 = slice1

        # load the slice of data from .labels
        self.load_slice()

        # fill the vector of simulated data for .counts
        if run:
            # infer count matrices on slice of data    
            self.run()

            # normalize counts while in stacked format?
            pass

            # get more features from the counts and flatten to .vector
            self.add_features()



    def add_features(self):
        """
        compute additional features that capture the stacked matrix structure
        of the count data before flattening it into a vector for ML.
        """
        # compute SVD features for each stack
        # (10, 15, 16, 16) -> (10, 15, 16, 16), (10, 15, 16), (10, 15, 16, 16)
        u, s, v = np.linalg.svd(self.counts)
        self.svdu = u
        self.svds = s
        self.svdv = v

        # compute variance (10, 15, 16, 16) -> (10, 16, 16)
        self.mvar = self.counts.var(axis=1)

        # reshape to ntests flattened (10, 15, 16, 16) -> (10, 3840)
        # vectorsnps = self.counts.reshape(self.counts.shape[0], -1)

        # return with stored vector results (10, ...)
        # self.vector = np.concatenate([
        #     vectorsnps,                              # snp counts 
        #     u.reshape(u.shape[0], -1),               # left singulars
        #     s.reshape(s.shape[0], -1),               # singulars 
        #     vh.reshape(vh.shape[0], -1),             # right singulars
        #     mvar.reshape(mvar.shape[0], -1),         # variances
        # ], axis=1)

        # compute ABBA, BABA and Hils statistic ratios...
        # ...


    def load_slice(self):
        """
        Pull data from .labels for use in ipcoal sims
        """
        # open view to the data
        with h5py.File(self.database, 'r') as io5:

            # sliced data arrays
            self.node_Nes = io5["node_Nes"][self.slice0:self.slice1, ...]
            self.admixture = io5["admixture"][self.slice0:self.slice1, ...]
            self.slide_seeds = io5["slide_seeds"][self.slice0:self.slice1]

            # attribute metadata
            self.tree = toytree.tree(io5.attrs["tree"])
            self.nsnps = io5.attrs["nsnps"]
            self.ntips = len(self.tree)

            # storage SNPs in STACKED matrix so we can compute stacked stats
            self.nquarts = sum(
                1 for i in itertools.combinations(range(self.ntips), 4))
            self.nvalues = self.slice1 - self.slice0
            self.counts = np.zeros(
                (self.nvalues, self.nquarts, 16, 16), dtype=np.int64) 



    def run(self):
        """
        iterate through ipcoal simulations across label values.
        """
        # run simulations
        for idx in range(self.nvalues):

            # modify the tree if ...
            tree = self.tree.mod.node_slider(
                prop=0.25, seed=self.slide_seeds[idx])

            # set Nes default and override on internal nodes with stored vals
            tree = tree.set_node_values("Ne", default=1e5)
            nes = iter(self.node_Nes[idx])
            for node in tree.treenode.traverse():
                if not node.is_leaf():
                    node.Ne = next(nes)

            # get admixture tuples (only supports 1 edge like this right now)
            admix = (
                int(self.admixture[idx, 0]),
                int(self.admixture[idx, 1]),
                0.5,
                self.admixture[idx, 2],
            )

            # build ipcoal Model object
            model = ipcoal.Model(
                tree=tree,
                admixture_edges=[admix],
                Ne=None,
                )

            # simulate genealogies and snps
            model.sim_snps(self.nsnps)

            # TODO: ipcoal converter not fastest possible
            mat = get_snps_count_matrix(tree, model.seqs)

            # store results
            self.counts[idx] = mat
