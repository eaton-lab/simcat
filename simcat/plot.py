#!/usr/bin/env python


import itertools
import numpy as np
import toyplot

from .utils import get_all_admix_edges, INVARIANTS, SimcatError
from .utils import ABBA_IDX, BABA_IDX, FIXED_IDX

# a series of colormaps 
PALETTE = toyplot.color.brewer.palette("Set2", count=5)
COLORMAPS = [
    toyplot.color.LinearMap(
        toyplot.color.Palette([
            toyplot.color.to_css(PALETTE[i]).rsplit(",", 1)[0] + ",0.000)",
            toyplot.color.to_css(PALETTE[i])
        ]),
        domain_min=0, domain_max=1,
    ) for i in range(5)
]
DIVERGING = toyplot.color.LinearMap(toyplot.color.brewer.palette("BlueRed"))



# # if it is a model object
# if hasattr(data, "nreps"):
#     if data.ntests > 1:
#         raise SimcatError("Cannot plot model with multiple tests.")
#     if data.nreps > 1:
#         counts = data.counts.mean(axis=0)
#     else:
#         counts = data.counts[0]
# elif isinstance(data, np.ndarray):
#     if len(data.shape) > 3:
#         counts = data.mean(axis=0)
#     else:
#         counts = data[0]
# else:
#     raise SimcatError("data type not recognized")


def draw_count_matrix(
    matrix=None, 
    normalize=True,
    mask_fixed_values=True, 
    show_invariants=True,
    **kwargs):
    """
    plot a count matrix with invariants sites overlaid.

    Parameters:
    -----------
    data: ndarray
        The array to be colormapped

    norm_variants: bool
        Normalize values to max site that is not invariant (e.g., AAAA)
    
    kwargs: dict
        Arguments to toyplot.canvas

    Useful kwarg options:
    ---------------------
    height, width, font-size.
    """

    # if no matrix then make empty
    if matrix is None:
        counts = np.zeros((16, 16), dtype=int)   
    else:
        counts = matrix

    # check shape
    if counts.shape != (16, 16):
        raise SimcatError("matrix shape must be (16, 16)")

    # set invariant cells to max
    if mask_fixed_values:
        counts = counts.copy()
        for idx in FIXED_IDX:
            counts[idx] = 0
        
    # normalize counts
    if normalize and (matrix is not None):
        counts = counts / counts.max()

    # if no user provided colormap use the gradient color[0]
    if not isinstance(counts, (tuple, list)):
        counts = (counts, COLORMAPS[0])

    # canvas arguments
    ckwargs = {
        "height": 750,
        "width": 750,
    }
    ckwargs.update({i: j for i, j in kwargs.items() if i in ckwargs})

    # table style arguments
    skwargs = {
        "font-size": "10px",
    }
    skwargs.update({i: j for i, j in kwargs.items() if i in skwargs})

    # create canvas and table
    canvas = toyplot.Canvas(**ckwargs)
    table = canvas.matrix(counts, margin=20)

    # style the table
    table.cells.grid.hlines[2:-2, 2:-2] = "single"
    table.cells.grid.vlines[2:-2, 2:-2] = "single"
    table.cells.cell[2:-2, 2:-2].data = INVARIANTS

    if show_invariants:
        table.body.cell[...].format = (
            toyplot.format.FloatFormatter(format="{:.2g}"))
    table.body.cell[...].lstyle = skwargs

    # fill hidden cells with grey
    if normalize:
        table.body.cell[0, 0].style = {"fill": "lightgrey"}
        table.body.cell[5, 5].style = {"fill": "lightgrey"}
        table.body.cell[10, 10].style = {"fill": "lightgrey"}
        table.body.cell[15, 15].style = {"fill": "lightgrey"}    

    # if no matrix then color ABBA BABA
    if matrix is None:
        for pair in BABA_IDX:
            table.body.cell[pair].style = {"fill": toyplot.color.Palette()[0]}
        for pair in ABBA_IDX:
            table.body.cell[pair].style = {"fill": toyplot.color.Palette()[1]}

    return canvas, table



def draw_five_matrix_comparison(model1, model2, domain_min=None, domain_max=None, **kwargs):
    """
    Plot tree, admixture and quartet matrices for the 5-taxon example.
    """

    # check for ntests, only allowed nreps here.
    if (model1.ntests > 1) or (model2.ntests > 1):
        raise SimcatError("Cannot plot for objects with ntests > 1")

    if (model1.nreps > 1) or (model2.nreps > 1):
        counts1 = model1.counts.mean(axis=0)
        counts2 = model2.counts.mean(axis=0)
    else:
        counts1 = model1.counts[0]
        counts2 = model2.counts[0]

    # tree and admixture edges from models.
    ttree = model1.tree
    admixture_edges = list(model1.admixture_edges) + list(model2.admixture_edges)

    # get abs values
    diff = counts1.astype(np.int64) - counts2.astype(np.int64)

    # set colormap 
    colormap = toyplot.color.LinearMap(
        toyplot.color.brewer.palette("BlueRed"), 
        center=0.0,
        domain_max=(domain_max if domain_max else diff.max()),
        domain_min=(domain_min if domain_min else diff.min()),
    )

    # canvas argumnets
    ckwargs = {
        "width": 2600, 
        "height": 1000,
    }
    ckwargs.update({i: j for i, j in kwargs.items() if i in ckwargs})

    # tree styling
    tkwargs = {
        "tree_style": "d", 
        "orient": "down",
        "edge_style": {"stroke-width": 12},
    }
    tkwargs.update({i: j for i, j in kwargs.items() if i in tkwargs})

    # build canvas
    canvas = toyplot.Canvas(**ckwargs)

    # get quartet sets
    qsets = [set(i) for i in itertools.combinations(range(5), 4)]
    qfull = set(range(5))
    
    # get node coordinates
    coords = ttree.get_node_coordinates()
    edges = get_all_admix_edges(ttree, 0.5, 0.5)

    # add tree to canvas
    x = 50
    for idx in range(5):
        
        tipcols = np.array(["#262626"] * 5)
        notidx = list(qfull - qsets[idx])[0]
        
        # draw tree
        axes = canvas.cartesian(bounds=(x + 50, x + 450, 100, 400))
        ttree.draw(
            axes=axes, 
            edge_colors=ttree.get_edge_values_from_dict({
                tuple(ttree.get_tip_labels()): '#262626',
                str(notidx): "lightgrey",
            }),
            **tkwargs)

        # tip dots
        axes.scatterplot(
            range(5), [0] * 5, color=[i for i in PALETTE], size=10)
        axes.y.domain.min = 0
        axes.show = False
        
        # admixture edge
        if admixture_edges:
            for edge in admixture_edges:
                src, dest = edge[0], edge[1]                
                xsrc, xdest = coords[src][0], coords[dest][0]
                height = edges[(src, dest)][0]
                arrow = toyplot.marker.create(
                    shape=">",
                    size=25,
                    mstyle={"fill": "grey", "stroke": "none"},
                )
                axes.graph(
                    np.array([(0, 1)]),
                    vcoordinates=[(xdest, height), (xsrc, height)],
                    tmarker=arrow,
                    estyle={
                        "stroke-width": 7, 
                        "stroke": "grey",
                    }
                )
        
        # tip dots
        tipcols[notidx] = "grey"
        axes.scatterplot(
            range(5), [0] * 5,
            size=30,
            color=tipcols,
        )

        # draw matrix
        table = canvas.matrix(
            (diff[idx], colormap),
            lshow=0, 
            tshow=0, 
            margin=20, 
            bounds=(x, x + 500, 450, 950),
        )

        # update hover values of table
        for i in range(16):
            for j in range(16):
                table.body.cell[i, j].title = "{}: {:.3f}".format(
                    INVARIANTS[i, j], diff[idx][i, j])

        # shift on x axis to next plot
        x += 500
    return canvas



def draw_five_taxon_matrix(model, **kwargs):
    """
    Plot count matrices and tree model for simulation with 5 tips.

    Parameters:
    -----------
    model: ndarray
        The array to be colormapped
    
    kwargs: dict
        Arguments to toyplot.canvas

    Useful kwarg options:
    ---------------------
    height, width, font-size.
    """
    # get tree from model
    ttree = model.tree
    if len(ttree) > 5:
        raise SimcatError("This plotting function is only for 5 tip trees")
    
    # check for ntests, only allowed nreps here.
    if model.ntests > 1:
        raise SimcatError("Cannot plot for objects with ntests > 1")

    if (model.nreps > 1):
        counts = model.counts.mean(axis=0)
    else:
        counts = model.counts[0]

    # set colormap 
    colormap = toyplot.color.LinearMap(
        toyplot.color.brewer.palette("BlueRed"), 
        center=0.0,
        domain_max=counts.max(),
        domain_min=counts.min(),
    )

    # canvas argumnets
    ckwargs = {
        "width": 2600, 
        "height": 1000,
    }
    ckwargs.update({i: j for i, j in kwargs.items() if i in ckwargs})

    # tree styling
    tkwargs = {
        "tree_style": "d", 
        "orient": "down",
        "edge_style": {"stroke-width": 12},
    }
    tkwargs.update({i: j for i, j in kwargs.items() if i in tkwargs})

    # build canvas
    canvas = toyplot.Canvas(**ckwargs)

    # get quartet sets
    qsets = [set(i) for i in itertools.combinations(range(5), 4)]
    qfull = set(range(5))
    
    # get node coordinates
    coords = ttree.get_node_coordinates()
    edges = get_all_admix_edges(ttree, 0.5, 0.5)

    # add tree to canvas
    x = 50
    for idx in range(5):
        
        tipcols = np.array(["#262626"] * 5)
        notidx = list(qfull - qsets[idx])[0]
        
        # draw tree
        axes = canvas.cartesian(bounds=(x + 50, x + 450, 100, 400))
        ttree.draw(
            axes=axes, 
            edge_colors=ttree.get_edge_values_from_dict({
                tuple(ttree.get_tip_labels()): '#262626',
                str(notidx): "lightgrey",
            }),
            **tkwargs)

        # tip dots
        axes.scatterplot(
            range(5), [0] * 5, color=[i for i in PALETTE], size=10)
        axes.y.domain.min = 0
        axes.show = False
        
        # admixture edge
        if model.admixture_edges:
            for edge in model.admixture_edges:
                src, dest = edge[0], edge[1]                
                xsrc, xdest = coords[src][0], coords[dest][0]
                height = edges[(src, dest)][0]
                arrow = toyplot.marker.create(
                    shape=">",
                    size=25,
                    mstyle={"fill": "grey", "stroke": "none"},
                )
                axes.graph(
                    np.array([(0, 1)]),
                    vcoordinates=[(xdest, height), (xsrc, height)],
                    tmarker=arrow,
                    estyle={
                        "stroke-width": 7, 
                        "stroke": "grey",
                    }
                )
        
        # tip dots
        tipcols[notidx] = "grey"
        axes.scatterplot(
            range(5), [0] * 5,
            size=30,
            color=tipcols,
        )

        # draw matrix
        table = canvas.matrix(
            (counts[idx], colormap),
            lshow=0, 
            tshow=0, 
            margin=20, 
            bounds=(x, x + 500, 450, 950),
        )

        # update hover values of table
        for i in range(16):
            for j in range(16):
                table.body.cell[i, j].title = "{}: {:.3f}".format(
                    INVARIANTS[i, j], counts[idx][i, j])


        # shift on x axis to next plot
        x += 500
    return canvas



def _draw_five_taxon_data(ttree, admixture_edges, counts, baseline=None, **kwargs):
    """
    Plot tree, admixture and quartet matrices for the 5-taxon example.
    """
    # scale tree to root height=3
    ttree = ttree.mod.node_scale_root_height(3)

    # canvas argumnets
    ckwargs = {
        "width": 2600, 
        "height": 1000,
        "style": {"background-color": "black"},
    }
    ckwargs.update({i: j for i, j in kwargs.items() if i in ckwargs})

    # tree styling
    tkwargs = {
        "tree_style": "d", 
        "orient": "down",
        "edge_style": {"stroke-width": 12, "stroke": "white"},
    }
    tkwargs.update({i: j for i, j in kwargs.items() if i in tkwargs})

    # build canvas
    canvas = toyplot.Canvas(**ckwargs)

    # get quartet sets
    qsets = [set(i) for i in itertools.combinations(range(5), 4)]
    qfull = set(range(5))
    
    # get node coordinates
    coords = ttree.get_node_coordinates()
    edges = get_all_admix_edges(ttree, 0.5, 0.5)

    # add tree to canvas
    x = 50
    for idx in range(5):
        
        tipcols = np.array(["white"] * 5)
        notidx = list(qfull - qsets[idx])[0]
        
        # draw tree
        axes = canvas.cartesian(bounds=(x + 50, x + 450, 100, 400))
        ttree.draw(
            axes=axes, 
            edge_colors=ttree.get_edge_values_from_dict({
                tuple(ttree.get_tip_labels()): 'white',
                str(notidx): "grey",
            }),
            **tkwargs)
        axes.scatterplot(
            range(5), [-0.5] * 5, color=[i for i in PALETTE], size=20)
        axes.y.domain.min = 0
        axes.show = False
        
        # admixture edge
        if admixture_edges:
            for edge in admixture_edges:
                src, dest = edge[0], edge[1]                
                xsrc, xdest = coords[src][0], coords[dest][0]
                height = edges[(src, dest)][0]
                arrow = toyplot.marker.create(
                    shape=">",
                    size=25,
                    mstyle={"fill": "white", "stroke": "none"},
                )
                axes.graph(
                    np.array([(0, 1)]),
                    vcoordinates=[(xdest, height), (xsrc, height)],
                    tmarker=arrow,
                    estyle={
                        "stroke-width": 7, 
                        #"stroke-dasharray": "5,5", 
                        "stroke": "white",
                    }
                )
        
        # tip dots
        tipcols[notidx] = "grey"
        axes.scatterplot(
            range(5), [0] * 5,
            size=30,
            color=tipcols,
        )

        if baseline:
            counts = abs(counts - baseline)
        
        # draw matrix
        canvas.matrix(
            (counts[idx], COLORMAPS[idx]), 
            lshow=0, 
            tshow=0, 
            margin=20, 
            bounds=(x, x + 500, 450, 950),
        )
        x += 500
    return canvas
