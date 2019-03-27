#!/usr/bin/env python

"""
Generate large database of site counts from coalescent simulations
based on msprime + toytree for using in machine learning algorithms.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

# imports
import sys
import toyplot
import toytree
import numpy as np
import msprime as ms
import itertools as itt
from scipy.special import comb

from .jitted import count_matrix_int, mutate_jc
from .utils import get_all_admix_edges, SimcatError



class Model:
    """
    A coalescent model for returning ms simulations.
    """
    def __init__(
        self, 
        tree,
        admixture_edges=None,
        admixture_type=0,
        theta=0.1,
        nsnps=5000,
        ntests=1,
        nreps=1,
        seed=None,
        debug=False,
        run=False,
        ):
        """
        An object used for demonstration and testing only. The real simulations
        use the similar object Simulator.

        Takes an input topology with edge lengths in coalescent units (2N) 
        entered as either a newick string or as a Toytree object,
        and generates 'ntests' parameter sets for running msprime simulations 
        which are stored in the the '.test_values' dictionary. The .run() 
        command can be used to execute simulations to fill count matrices 
        stored in .counts. Admixture events (intervals or pulses) from source
        to dest are described as viewed backwards in time. 

        Parameters:
        -----------
        tree: (str)
            A newick string or Toytree object of a species tree with edges in
            coalescent units.

        admixture_edges (list, tuple):
            A list of admixture events in the 'admixture interval' format:           
            (source, dest, (edge_min, edge_max), (rate_min, rate_max)). 
            e.g., (3, 5, 0.5, 0.01)
            e.g., (3, 5, (0.5, 0.5), (0.05, 0.5))
            e.g., (1, 3, (0.1, 0.9), 0.05)
            The source sends migrants to destination **backwards in time.**
            The edge min, max are *proportions* of the length of the edge that 
            overlaps between source and dest edges over which admixture can
            occur. If None then default values of 0.25 and 0.75 are used, 
            meaning introgression can occur over the middle 50% of the edge. 
            The rate min, max are migration rates or proportions that will be
            either a single value or sampled from a range. For 'rate' details 
            see the 'admixture type' parameter.

        admixture_type (str, int):
            Either "pulsed" (0; default) or "interval" (1). 
            If "pulsed" then admixture occurs at a single time point
            selected uniformly from the admixture interval (e.g., (0.1, 0.9) 
            can select over 90% of the overlapping edge; (0.5, 0.5) would only
            allow admixture at the midpoint). The 'rate' parameter is the 
            proportion of one population that will be introgressed into the 
            other. 
            If "interval" then admixture occurs uniformly over the entire 
            admixture interval and 'rate' is a constant migration rate over 
            this time period.

        theta (float, tuple):
            Mutation parameter. Enter a float, or a tuple of floats to supply
            a range to sample from over ntests. If None then values will be
            extracted from the Toytree if it has a 'theta' feature on each 
            internal node. Else an errror will be raise if no thetas found.

        nsnps (int):
            Number of unlinked SNPs simulated (e.g., counts is (nsnps, 16, 16))

        ntests (int):
            Number of parameter sets to sample for each event, i.e., given 
            a theta range and admixture events range multiple sets of parameter
            values could be sampled. The counts array is expanded to be 
            (ntests, nsnps, 16, 16)

        nreps (int):
            Number of technical replicates to run using the same param sets.
            The counts array is expanded to be (nreps * ntests, nsnps, 16, 16)

        seed (int):
            Random number generator
        """
        # init random seed
        np.random.seed(seed)

        # hidden argument to turn on debugging
        self._debug = debug

        # store sim params: fixed mut; range theta; Ne computed from theta,mut
        theta = ((theta,) if isinstance(theta, (int, float)) else theta)
        self.theta = np.array((min(theta), max(theta)))
        self.mut = 1e-7

        # dimension of simulations
        self.nsnps = nsnps
        self.ntests = ntests
        self.nreps = nreps

        # parse the input tree
        if isinstance(tree, toytree.Toytree.ToyTree):
            self.tree = tree
        elif isinstance(tree, str):
            self.tree = toytree.tree(tree)
        else:
            raise TypeError("input tree must be newick str or Toytree object")
        self.ntips = len(self.tree)

        ## storage for output
        self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb
        self.counts = np.zeros(
            (self.ntests * self.nreps, self.nquarts, 16, 16), dtype=np.int64)

        # store node.name as node.idx, save old names in a dict.
        self.namedict = {}
        for node in self.tree.treenode.traverse():
            if node.is_leaf():
                # store old name and set new one
                self.namedict[str(node.idx)] = node.name
                node.name = str(node.idx)

        # check formats of admixture args
        self.admixture_edges = (admixture_edges if admixture_edges else [])
        self.admixture_type = (1 if admixture_type in (1, "interval") else 0)
        if self.admixture_edges:
            if not isinstance(self.admixture_edges[0], (list, tuple)):
                self.admixture_edges = [self.admixture_edges]
            for edge in self.admixture_edges:
                if len(edge) != 5:
                    raise ValueError(
                        "admixture edges should each be a tuple with 5 values")
        self.aedges = (0 if not self.admixture_edges else len(self.admixture_edges))

        # generate sim parameters from the tree and admixture scenarios
        # stores in self.sims 'mrates' and 'mtimes'
        self._get_test_values()

        # fill the counts matrix or call run later
        if run:
            self.run()


    @property
    def Ne(self):
        "Ne is calculated from theta and fixed mut (theta=4Neu)"
        return 


    def _get_test_values(self): 
        """
        Generates mrates, mtimes, and thetas arrays for simulations. 

        Migration times are uniformly sampled between start and end points that
        are constrained by the overlap in edge lengths, which is automatically
        inferred from 'get_all_admix_edges()'. migration rates are drawn 
        uniformly between 0.0 and 0.5. thetas are drawn uniformly between 
        theta0 and theta1, and Ne is just theta divided by a constant. 

        self.test_values = {
            thetas: [1, 2, 0.2, .1, .5], 
            1: {mrates: [.5, .2, .3], mtimes: [(2, 3), (4, 5), (1, 2)]}, 
            2: {mrates: [.01, .05,], mtimes: [(0.5, None), 0.1, None)]
            3: {...}
            ...
        }
        """
        # dictionary to store arrays of params for each admixture scenario
        self.test_values = {
            "thetas": np.random.uniform(
                low=self.theta[0], high=self.theta[1], size=self.ntests), 
        }

        # sample times and proportions/rates for admixture intervals
        idx = 0
        for iedge in self.admixture_edges:

            # use rate/prop if provided else sample from exponential
            if iedge[4]:
                mrates = np.repeat(iedge[4], self.ntests)
                mo = (iedge[4], iedge[4])
            
            else:                
                # sample migration rates in range 1/1000 to 1/10 per gen
                if self.admixture_type:
                    mo = (0.0, 0.1)
                    mrates = np.random.uniform(*mo, size=self.ntests)
                # sample migration pulse in range 5% to 50% of population
                else:
                    mo = (0.05, 0.5)
                    mrates = np.random.uniform(*mo, size=self.ntests)

            # intervals are overlapping edges where admixture can occur. 
            # lower and upper restrict the range along intervals for each 
            intervals = get_all_admix_edges(self.tree, iedge[2], iedge[3])
            snode = self.tree.treenode.search_nodes(idx=iedge[0])[0]
            dnode = self.tree.treenode.search_nodes(idx=iedge[1])[0]
            ival = intervals[snode.idx, dnode.idx]

            # intervals mode
            if self.admixture_type:
                ui = np.random.uniform(ival[0], ival[1], self.ntests * 2)
                ui = ui.reshape((self.ntests, 2))
                mtimes = np.sort(ui, axis=1)  
            # pulsed mode
            else:
                ui = np.random.uniform(ival[0], ival[1], self.ntests)
                null = np.repeat(None, self.ntests)
                mtimes = np.stack((ui, null), axis=1)

            # store values only if migration is high enough to be detectable
            self.test_values[idx] = {
                "mrates": mrates, 
                "mtimes": mtimes,
            }
            idx += 1

            # print info
            if self._debug:
                print("migration: edge({}->{}) time({:.3f}, {:.3f}), rate({:.3f}, {:.3f})"
                    .format(snode.idx, dnode.idx, ival[0], ival[1], mo[0], mo[1]),
                    file=sys.stderr)


    def _get_demography(self):
        """
        returns demography scenario based on an input tree and admixture
        edge list with events in the format (source, dest, start, end, rate).
        Time on the tree is defined in coalescent units, which here is 
        converted to time in 2Ne generations as an int.
        """
        ## Define demographic events for msprime
        demog = set()

        ## tag min index child for each node, since at the time the node is 
        ## called it may already be renamed by its child index b/c of 
        ## divergence events.
        for node in self.tree.treenode.traverse():
            if node.children:
                node._schild = min([i.idx for i in node.get_descendants()])
            else:
                node._schild = node.idx

        ## Add divergence events (converts time to N generations)
        for node in self.tree.treenode.traverse():
            if node.children:
                dest = min([i._schild for i in node.children])
                source = max([i._schild for i in node.children])
                time = int(node.height * 2. * self._Ne)
                demog.add(ms.MassMigration(time, source, dest))
                if self._debug:
                    print('div time: {} {} {}'
                        .format(int(time), source, dest), file=sys.stderr)

        ## Add migration pulses
        if not self.admixture_type:
            for evt in range(self.aedges):
                rate = self._mrates[evt]
                time = int(self._mtimes[evt][0] * 2. * self._Ne)
                source, dest = self.admixture_edges[evt][:2]

                ## rename nodes at time of admix in case divergences renamed them
                snode = self.tree.treenode.search_nodes(idx=source)[0]
                dnode = self.tree.treenode.search_nodes(idx=dest)[0]
                children = (snode._schild, dnode._schild)
                demog.add(ms.MassMigration(time, children[0], children[1], rate))
                if self._debug:
                    print('mig pulse: {} ({:.3f}), {}, {}, {:.3f}'.format(
                        time, self._mtimes[evt][0], source, dest, rate),
                        file=sys.stderr)

        ## Add migration intervals
        else:
            for evt in range(self.aedges):
                rate = self._mrates[evt]
                time = (self._mtimes[evt] * 2. * self._Ne).astype(int)
                source, dest = self.admixture_edges[evt][:2]

                ## rename nodes at time of admix in case divergences renamed them
                snode = self.tree.treenode.search_nodes(idx=source)[0]
                dnode = self.tree.treenode.search_nodes(idx=dest)[0]
                children = (snode._schild, dnode._schild)
                demog.add(ms.MigrationRateChange(time[0], rate, children))
                demog.add(ms.MigrationRateChange(time[1], 0, children))
                if self._debug:
                    print("mig interv: {}, {}, {}, {}, {:.3f}".format(
                        time[0], time[1], children[0], children[1], rate),
                        file=sys.stderr)

        ## sort events by time
        demog = sorted(list(demog), key=lambda x: x.time)
        return demog


    def _get_popconfig(self):
        """
        returns population_configurations for N tips of a tree
        """
        population_configurations = [
            ms.PopulationConfiguration(sample_size=1, initial_size=self._Ne)
            for ntip in range(self.ntips)]
        return population_configurations


    def _get_tree_sequence(self, idx):
        """
        Performs simulations with params varied across input values.
        """       
        # migration scenarios from admixture_edges, used in demography
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()
        self._mtimes = [
            self.test_values[evt]['mtimes'][idx] for evt in 
            range(len(self.admixture_edges))
        ] 
        self._mrates = [
            self.test_values[evt]['mrates'][idx] for evt in 
            range(len(self.admixture_edges))
        ]

        # Ne is calculated from fixed mut and sampled theta. Used in popconfig
        self._theta = self.test_values["thetas"][idx]
        self._Ne = int((self._theta / self.mut) / 4.)

        # debug printer
        if self._debug:
            print("pop: Ne:{}, theta:{:.3f}, mut:{:.2E}"
                .format(self._Ne, self._theta, self.mut),
                file=sys.stderr)

        # msprime simulation to make tree_sequence generator
        sim = ms.simulate(
            migration_matrix=migmat,
            num_replicates=self.nsnps * 1000,                 # ensures SNPs 
            population_configurations=self._get_popconfig(),  # applies Ne
            demographic_events=self._get_demography(),        # applies popst. 
        )
        return sim


    def run(self):
        """
        run and parse results for nsamples simulations.
        """
        # iterate over ntests (different sampled simulation parameters)
        gidx = 0
        for ridx in range(self.ntests):
            
            # get tree_sequence generator for this set of params
            sims = self._get_tree_sequence(ridx)

            # repeat draws from this generator for each technical rep
            for rep in range(self.nreps):

                # store results (nsnps, ntips); def. 1000 SNPs
                snparr = np.zeros((self.nsnps, self.ntips), dtype=np.int64)

                # continue until all SNPs are sampled from generator
                nsnps = 0
                nfail = 0
                while nsnps < self.nsnps:

                    # get next tree and drop mutations 
                    muts = ms.mutate(next(sims), rate=self.mut)
                    bingenos = muts.genotype_matrix()

                    # convert binary SNPs to {0,1,2,3} using JC 
                    if bingenos.size:
                        sitegenos = mutate_jc(bingenos, self.ntips)
                        snparr[nsnps] = sitegenos
                        nsnps += 1
                    else:
                        nfail += 1

                if self._debug:
                    print("{} trees to get {} with mutations"
                        .format(nsnps + nfail, self.nsnps), file=sys.stderr)

                # organize SNPs array into multiple 16x16 arrays
                # iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
                quartidx = 0
                qiter = itt.combinations(range(self.ntips), 4)
                for currquart in qiter:
                    # cols indices match tip labels b/c we named tips node.idx
                    quartsnps = snparr[:, currquart]
                    self.counts[gidx, quartidx] = count_matrix_int(quartsnps)                    
                    # self.counts[gidx, quartidx] = count_matrix_float(quartsnps)
                    quartidx += 1

                # scale by max count for this rep
                # self.counts[gidx, ...] /= self.counts[gidx, ...].max()
                gidx += 1

            if self._debug: 
                print("\n", file=sys.stderr)


    def plot_test_values(self):
        """
        Returns a toyplot canvas 
        """
        # canvas, axes = plot_test_values(self.tree)
        if not self.counts.sum():
            raise SimcatError("No mutations generated. First call '.run()'")

        ## setup canvas
        canvas = toyplot.Canvas(height=250, width=800)

        ax0 = canvas.cartesian(
            grid=(1, 3, 0))
        ax1 = canvas.cartesian(
            grid=(1, 3, 1), 
            xlabel="simulation index",
            ylabel="migration intervals", 
            ymin=0, 
            ymax=self.tree.treenode.height)  # * 2 * self._Ne)
        ax2 = canvas.cartesian(
            grid=(1, 3, 2), 
            xlabel="proportion migrants", 
            #xlabel="N migrants (M)", 
            ylabel="frequency")

        ## advance colors for different edges starting from 1
        colors = iter(toyplot.color.Palette())

        ## draw tree
        self.tree.draw(
            tree_style='c', 
            node_labels=self.tree.get_node_values("idx", 1, 1),
            tip_labels=False, 
            axes=ax0,
            node_sizes=16,
            padding=50)
        ax0.show = False

        # iterate over edges 
        for tidx in range(self.aedges):
            color = next(colors)

            ## get values for the first admixture edge
            mtimes = self.test_values[tidx]["mtimes"]
            mrates = self.test_values[tidx]["mrates"]
            mt = mtimes[mtimes[:, 0].argsort()]
            boundaries = np.column_stack((mt[:, 0], mt[:, 1]))

            ## plot
            for idx in range(boundaries.shape[0]):
                ax1.fill(
                    #boundaries[idx],
                    (boundaries[idx][0], boundaries[idx][0] + 0.1),                    
                    (idx, idx),
                    (idx + 0.5, idx + 0.5),
                    along='y',
                    color=color, 
                    opacity=0.5)

            # migration rates/props
            ax2.bars(
                np.histogram(mrates, bins=20), 
                color=color, 
                opacity=0.5,
            )

        return canvas, (ax0, ax1, ax2)

