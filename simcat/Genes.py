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
from copy import deepcopy
from scipy import linalg

from .jitted import count_matrix_int, mutate_jc, base_to_int_genes


#############################################################################
class Genes:
    """ 
    This object runs msprime on a species tree with parameters that we supply
    """
    def __init__(self, database_file, num_genes, gene_length,slice0,slice1,mutator):
       
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

            self.num_genes = num_genes
            self.gene_length = gene_length

            # attribute metadata
            self.tree = toytree.tree(io5.attrs["tree"])
            self.ntips = len(self.tree)
            self.aedges = self.asources.shape[1]

            # storage for output
            self.alignment_length = num_genes*gene_length
            self.nvalues = self.slice1 - self.slice0
            self.alignment = np.zeros(
                (self.nvalues, self.ntips,self.alignment_length), dtype=np.int8) 


    def _get_tree_sequence(self, idx):
        """
        Performs simulations with params varied across input values.
        """       
        # Ne is calculated from fixed mut and sampled theta. Used in popconfig
        self._theta = self.thetas[idx]
        self._Ne = int((self._theta / self.mut) / 4.)
        print(self._Ne)
        self._atimes = self.atimes[idx]
        self._aprops = self.aprops[idx]
        self._asources = self.asources[idx]
        self._atargets = self.atargets[idx]

        # msprime simulation to make tree_sequence generator
        sim = ms.simulate(
            # mutation_rate=self.mut,
            length=self.gene_length,
            num_replicates=self.num_genes*10000,                # ensures SNPs 
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

    def mutate_jc(self, ttree, seq_size,
              return_tips = True,
              q=np.array([
                            [-1.,(1./3),(1./3),(1./3)],
                            [(1./3),-1.,(1./3),(1./3)],
                            [(1./3),(1./3),-1.,(1./3)],
                            [(1./3),(1./3),(1./3),-1.],
                        ]),
              basearr = np.array([0,1,2,3])
             ):
        ## dictionary to hold the evolving sequences at each node
        sim_dict = dict()
        
        ## for each node...
        for curr_node in ttree.treenode.traverse():
            ## if not the root...
            if not curr_node.is_root():
                ## take the sequence from the previous node
                seq=sim_dict[curr_node.up.idx]
                
                ## copy that sequence into a "new sequence" object
                newseq = deepcopy(seq)
                
                ## from q matrix and branch lenght, get the probability of observing each state (our P matrix)
                probs = linalg.expm(np.multiply(q,float(curr_node.dist)))
                
                ## cheating here... basically ask "which of these bases change at all?"
                ## for JC *specifically*, we can just pull the value in the top left corner of the matrix
                ## and use it for all bases (ie, equal probability of A changing to something else as the prob
                ## of G changing to something else)
                changers = (1-np.random.binomial(1,probs[0][0],size=len(newseq))).astype(bool)
                
                ## pull out values that are changing
                mask = newseq[changers]
                
                ## make an empty array to hold the values to which they are changing
                newarr = np.zeros(mask.shape,dtype=np.int8)
                
                ## for each value that is changing to something else, randomly draw one of the 
                ## other three bases. Again, this is JC SPECIFIC
                for i in range(len(mask)):
                    newarr[i] = np.random.choice(np.delete(basearr, mask[i]))
                
                ## plug the changed values back into the newseq object
                newseq[changers]=newarr
                ## save the new sequence to its assigned node
                sim_dict[curr_node.idx] = newseq
            

            else:
                # if the root, just generate a random sequence
                sim_dict[curr_node.idx] = np.random.choice(4,size=seq_size)

        if return_tips:
    ##        tip_dict = dict()
    ##        for leaf in range(len(ttree)):
    ##            tip_dict[leaf] = sim_dict[leaf]
            tip_dict = {k: sim_dict[k] for k in [leaf.idx for leaf in ttree.treenode.get_leaves()]}
            return(tip_dict)
        else:
            return(sim_dict)

    def run(self):
        """
        run and parse results for nsamples simulations.
        """
        # iterate over ntests (different sampled simulation parameters)
        for idx in range(self.nvalues):

            # temporarily format these as stacked matrices
            #tmpcounts = np.zeros((self.nquarts,16,16),dtype= np.int64)

            # get tree_sequence for this set
            sims = self._get_tree_sequence(idx)

            # store results (nsnps, ntips); def. 1000 SNPs
            snparr = np.zeros((self.ntips, self.alignment_length), dtype=np.int64)

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
                num_finished_genes = 0
                num_failures = 0
                while num_finished_genes < self.num_genes: 
                	try:
	                    new_treeseq = next(sims).trees()
	                    curr_bp = 0 # track what basepair we're on
	                    for gt in new_treeseq:

	                        gt_start = gt.interval[0]
	                        gt_end = gt.interval[1]
	                        gt_len = int(round(gt_end)) - int(round(gt_start))
	                        #print(newtree)
	                        #filename = str(np.random.randint(1e10)) +'.newick'
	                        #with open(filename,'w') as f:
	                        #    f.write(str(newtree))
	                        #process = Popen(['seq-gen', '-m','GTR','-l','1','-s',str(self.mut),filename,'-or','-q'], stdout=PIPE, stderr=PIPE)
	                        #stdout, stderr = process.communicate()
	                        #result=stdout.decode("utf-8").split('\n')[:-1]
	                        #geno = dict([i.split(' ') for i in result[1:]])

	                        newick = gt.newick()

	                        my_partition = pyvolve.Partition(models = my_model, size = gt_len)
	                        t = pyvolve.read_tree(tree = newick,scale_tree = self.mut)
	                        my_evolver = pyvolve.Evolver(partitions = my_partition, tree = t)
	                        my_evolver(seqfile=None)
	                        geno=my_evolver.leaf_seqs
	                        ordered = [geno[np.str(i)] for i in range(1,len(geno)+1)]
	                        #snparr[nsnps] = base_to_int(ordered)
	                        #if os.path.isfile(filename):
	                        #    os.remove(filename)
	                        #else:    ## Show an error ##
	                        #    print("Error: %s file not found" % filename)
	                        snparr[num_finished_genes,:,curr_bp:(curr_bp+gt_len)] = base_to_int_genes(np.array(ordered))
	                        curr_bp += gt_len
	                    num_finished_genes += 1
	                except LibraryError:
	                	num_failures += 1
	                	print("failed: "+ str(num_failures))
		                pass
                return(snparr)

            elif self.mutator == 'toytree':
                while nsnps < self.nsnps:
                    newtree = toytree.tree(next(next(sims).trees()).newick())
                    with open('newick.tre','w') as f:
                        f.write(newtree.write(tree_format=5))
                        f.write(newtree.mod.node_scale_root_height(newtree.treenode.height*self.mut).write(tree_format=5))
                    seq = SeqGen(
                        newtree.mod.node_scale_root_height(newtree.treenode.height*self.mut),
                        model="JC",
                        #seed=123,
                    )
                    geno=seq.mutate(1)
                    ordered = [geno[np.str(i)] for i in range(1,len(geno)+1)]
                    if len(np.unique(ordered)) > 1:
                        snparr[nsnps] = base_to_int(ordered)
                        nsnps += 1

            elif self.mutator == 'jc':
                while nsnps < self.nsnps:
                    try:
                        nextone = next(sims).trees()
                        newick = next(nextone).newick()
                        newtree = toytree.tree(newick)
                        #with open('newick.tre','w') as f:
                        #    f.write(newtree.write(tree_format=5))
                        #    f.write(newtree.mod.node_scale_root_height(newtree.treenode.height*self.mut).write(tree_format=5))
                        #seq = SeqGen(
                        #    newtree.mod.node_scale_root_height(newtree.treenode.height*self.mut),
                        #    model="JC",
                        #    #seed=123,
                        #)



                        geno=self.mutate_jc(
                            newtree.mod.node_scale_root_height(newtree.treenode.height*self.mut),
                            1
                            )
                        ordered = [geno[i] for i in range(0,len(geno))]
                        if len(np.unique(ordered)) > 1:
                            snparr[nsnps] = ordered
                            nsnps += 1

                    # This can occur when pop size is v small, just skip to next.
                    except:
                        pass


