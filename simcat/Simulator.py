#!/usr/bin/env python

"""
Pulls a slice from the database and runs simulation to get SNP counts.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

import h5py
import toytree
from toytree.SeqGen import SeqGen
import numpy as np
import msprime as ms
import itertools as itt
from scipy.special import comb
from _msprime import LibraryError
from subprocess import Popen, PIPE
import os
import pyvolve

from .jitted import count_matrix_int, mutate_jc, base_to_int


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
        self.mut = 1e-7
        self.theta = None

        self.mutator = mutator

        # open view to the data
        with h5py.File(self.database, 'r') as io5:

            # sliced data arrays
            self.thetas = io5["thetas"][slice0:slice1]
            self.atimes = io5["admix_times"][slice0:slice1, ...]
            self.asources = io5["admix_sources"][slice0:slice1, ...]
            self.atargets = io5["admix_targets"][slice0:slice1, ...]
            self.aprops = io5["admix_props"][slice0:slice1, ...]

            # attribute metadata
            self.tree = toytree.tree(io5.attrs["tree"])
            self.nsnps = io5.attrs["nsnps"]
            self.ntips = len(self.tree)
            self.aedges = self.asources.shape[1]

            # storage for output
            self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb
            self.nvalues = self.slice1 - self.slice0
            self.counts = np.zeros(
                (self.nvalues, self.nquarts*16*16), dtype=np.int64) 

        # calls run and returns filled counts matrix
        if run:
            self.run()


    def _get_tree_sequence(self, idx):
        """
        Performs simulations with params varied across input values.
        """       
        # Ne is calculated from fixed mut and sampled theta. Used in popconfig
        self._theta = self.thetas[idx]
        self._Ne = int((self._theta / self.mut) / 4.)
        self._atimes = self.atimes[idx]
        self._aprops = self.aprops[idx]
        self._asources = self.asources[idx]
        self._atargets = self.atargets[idx]

        # msprime simulation to make tree_sequence generator
        sim = ms.simulate(
            # mutation_rate=self.mut,
            # length=self.length,
            num_replicates=self.nsnps * 10000,                # ensures SNPs 
            population_configurations=self._get_popconfig(),  # applies Ne
            demographic_events=self._get_demography(),        # applies popst. 
        )
        return sim



    def _get_popconfig(self):
        """
        returns population_configurations for N tips of a tree
        """
        population_configurations = [
            ms.PopulationConfiguration(sample_size=1, initial_size=self._Ne)
            for ntip in range(self.ntips)]
        return population_configurations



    def _get_demography(self):
        """
        returns demography scenario based on an input tree and admixture
        edge list with events in the format (source, dest, start, end, rate)
        """
        # Define demographic events for msprime
        demog = set()

        # tag min index child for each node, since at the time the node is 
        # called it may already be renamed by its child b/c of div events.
        for node in self.tree.treenode.traverse():
            if node.children:
                node._schild = min([i.idx for i in node.get_descendants()])
            else:
                node._schild = node.idx

        # Add divergence events (converts time to N generations)
        for node in self.tree.treenode.traverse():
            if node.children:
                dest = min([i._schild for i in node.children])
                source = max([i._schild for i in node.children])
                time = int(node.height * 2. * self._Ne)
                demog.add(ms.MassMigration(time, source, dest))

        # Add migration pulses
        for evt in range(self.aedges):
            rate = self._aprops[evt]
            time = int(self._atimes[evt] * 2. * self._Ne)
            source = self._asources[evt]
            dest = self._atargets[evt]

            # rename nodes at time of admix in case divergences renamed them
            snode = self.tree.treenode.search_nodes(idx=source)[0]
            dnode = self.tree.treenode.search_nodes(idx=dest)[0]
            children = (snode._schild, dnode._schild)
            demog.add(ms.MassMigration(time, children[0], children[1], rate))

        ## sort events by time
        demog = sorted(list(demog), key=lambda x: x.time)
        return demog



    def run(self):
        """
        run and parse results for nsamples simulations.
        """
        # iterate over ntests (different sampled simulation parameters)
        for idx in range(self.nvalues):

            # temporarily format these as stacked matrices
            tmpcounts = np.zeros((self.nquarts,16,16),dtype= np.int64)

            # get tree_sequence for this set
            sims = self._get_tree_sequence(idx)

            # store results (nsnps, ntips); def. 1000 SNPs
            snparr = np.zeros((self.nsnps, self.ntips), dtype=np.int64)

            # continue until all SNPs are sampled from generator
            nsnps = 0
            if self.mutator == 'msprime':
                while nsnps < self.nsnps:

                    # wrap for _msprime.LibraryError 
                    try:
                        # get next tree and drop mutations 
                        muts = ms.mutate(next(sims), rate=self.mut)
                        bingenos = muts.genotype_matrix()

                        # convert binary SNPs to {0,1,2,3} using JC 
                        if bingenos.size:
                            sitegenos = mutate_jc(bingenos, self.ntips)
                            snparr[nsnps] = sitegenos
                            nsnps += 1

                    # This can occur when pop size is v small, just skip to next.
                    except LibraryError:
                        pass

            elif self.mutator == 'pyvolve':
                my_model = pyvolve.Model('nucleotide')
                my_partition = pyvolve.Partition(models = my_model, size = 1)
                while nsnps < self.nsnps:
                    newtree = next(next(sims).trees()).newick()
                    #print(newtree)
                    #filename = str(np.random.randint(1e10)) +'.newick'
                    #with open(filename,'w') as f:
                    #    f.write(str(newtree))
                    #process = Popen(['seq-gen', '-m','GTR','-l','1','-s',str(self.mut),filename,'-or','-q'], stdout=PIPE, stderr=PIPE)
                    #stdout, stderr = process.communicate()
                    #result=stdout.decode("utf-8").split('\n')[:-1]
                    #geno = dict([i.split(' ') for i in result[1:]])

                    t = pyvolve.read_tree(tree = newtree,scale_tree = self.mut)
                    my_evolver = pyvolve.Evolver(partitions = my_partition, tree = t)
                    my_evolver(seqfile=None)
                    geno=my_evolver.leaf_seqs
                    ordered = [geno[np.str(i)] for i in range(1,len(geno)+1)]
                    snparr[nsnps] = base_to_int(ordered)
                    #if os.path.isfile(filename):
                    #    os.remove(filename)
                    #else:    ## Show an error ##
                    #    print("Error: %s file not found" % filename)
                    nsnps += 1

            elif self.mutator == 'toytree':
                while nsnps < self.nsnps:
                    newtree = toytree.tree(next(next(sims).trees()).newick())
                    seq = SeqGen(
                        tree,
                        model="JC",
                        seed=123,
                    )
                    geno=seq.mutate(1)
                    ordered = [geno[np.str(i)] for i in range(1,len(geno)+1)]
                    snparr[nsnps] = base_to_int(ordered)
                    nsnps += 1

            # iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
            quartidx = 0
            qiter = itt.combinations(range(self.ntips), 4)
            for currquart in qiter:
                # cols indices match tip labels b/c we named tips node.idx
                quartsnps = snparr[:, currquart]
                # save as stacked matrices
                tmpcounts[quartidx] = count_matrix_int(quartsnps)
                # save flattened to counts
                self.counts[idx] = np.ravel(tmpcounts)
                quartidx += 1
