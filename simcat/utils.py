#!/usr/bin/env python

import sys
import time
import datetime
import numpy as np

from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display


class SimcatError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def get_all_admix_edges(ttree, lower=0.25, upper=0.25):
    """
    Find all possible admixture edges on a tree. Edges are unidirectional, 
    so the source and dest need to overlap in time interval. To retrict 
    migration to occur away from nodes (these can be harder to detect when 
    validating methods) you can set upper and lower limits. For example, to 
    make all source migrations to occur at the midpoint of overlapping 
    intervals in which migration can occur you can set upper=.5, lower=.5.   
    """
    # bounds on edge overlaps
    if lower is None:
        lower = 0.0
    if upper is None:
        upper = 1.0

    ## for all nodes map the potential admixture interval
    for snode in ttree.treenode.traverse():
        if snode.is_root():
            snode.interval = (None, None)
        else:
            snode.interval = (snode.height, snode.up.height)

    ## for all nodes find overlapping intervals
    intervals = {}
    for snode in ttree.treenode.traverse():
        for dnode in ttree.treenode.traverse():
            if not any([snode.is_root(), dnode.is_root(), dnode == snode]):
                ## check for overlap
                smin, smax = snode.interval
                dmin, dmax = dnode.interval

                ## find if nodes have interval where admixture can occur
                low_bin = np.max([smin, dmin])
                top_bin = np.min([smax, dmax])              
                if top_bin > low_bin:

                    # restrict migration within bin to a smaller interval
                    length = top_bin - low_bin
                    low_limit = low_bin + (length * lower)
                    top_limit = low_bin + (length * upper)
                    intervals[(snode.idx, dnode.idx)] = (low_limit, top_limit)
    return intervals



# def progress_bar(njobs, nfinished, start, message=""):
#     "prints a progress bar"
#     ## measure progress
#     if njobs:
#         progress = 100 * (nfinished / njobs)
#     else:
#         progress = 100

#     ## build the bar
#     hashes = "#" * int(progress / 5.)
#     nohash = " " * int(20 - len(hashes))

#     ## get time stamp
#     elapsed = datetime.timedelta(seconds=int(time.time() - start))

#     ## print to stderr
#     args = [hashes + nohash, int(progress), elapsed, message]
#     print("\r[{}] {:>3}% | {} | {}".format(*args), end="")
#     sys.stderr.flush()



class Progress(object):
    def __init__(self, njobs, message):
        self.njobs = njobs
        self.message = message
        self.start = time.time()
        self.bar = IntProgress(
            value=0, min=0, max=self.njobs, 
            layout={"width": "350px"})
        self.label = HTML(self.printstr)
        self.widget = VBox(
            children=[self.label, self.bar], 
            layout={"padding": "0px", "margin": "0px"})
        
    @property
    def printstr(self):
        elapsed = datetime.timedelta(seconds=int(time.time() - self.start))
        s1 = "<span style='font-size:14px; font-family:monospace'>"
        s2 = "</span>"
        inner = "{} | {:>3}% | {}".format(
            self.message, 
            int(100 * (self.bar.value / self.njobs)),
            elapsed,
        )
        return s1 + inner + s2

    def display(self):
        display(self.widget)
    
    def increment_all(self):
        self.bar.value += 1
        if self.bar.value == self.njobs:
            self.bar.bar_style = "success"
        self.increment_time()
            
    def increment_time(self):
        self.label.value = self.printstr
