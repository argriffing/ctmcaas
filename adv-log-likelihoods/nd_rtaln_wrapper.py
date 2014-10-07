"""
Wrap rtaln.

Current models of evolution often assume that random variables have any
one of a finite number of values at any given time, and that the value
changes over continuous time in a way that depends only on its current value
and is independent of the values of other variables.

This is not always the case. For example, codon models assume that
nucleotide sites may evolve in a way that depends on the state of other
nucleotides in the same codon.

We want to add more of this kind of dependence into evolutionary models.
A straightforward way to do this is to combine multiple variables
into a single new variable representing the joint state of the component
variables. This combination allows us to represent dependent evolution
among components of the joint variable, while preserving the Markov
property of the joint variable and while preserving independence among
such combinations of variables.

This wrapper 'flattens' and 'unflattens' the state space of combined variables,
acting as a mediator between the detailed viewpoint that sees each
component separately and the broad viewpoint that sees the compound state
as a single variable. An example of the more detailed viewpoint could be of
an observer who records the state of only certain components at certain times.
An example of the broader viewpoint could be of a function in a transition
sampler that needs to know the expected waiting time before any transition
happens to any of the components of a random variable.

"""
#FIXME Converting the model from the advanced format used to compute log
#FIXME likelihoods to the format required by rtaln is not going to work
#FIXME until rtaln allows inhomogeneous processes, that is,
#FIXME different processes on different edges of the tree.

from __future__ import print_function, division

import random
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def reordered_nodes_and_edges(nodes, edges):
    """
    Find an order palatable to rtaln.

    The rtaln interface requires nodes to be ordered breadth first.
    The edges are ordered according to the index of the tail node of the edge.
    Ideally the host will call this function and use the recommended ordering,
    rather than having to also invert the ordering of the rtaln output.

    """
    T = nx.DiGraph()
    T.add_nodes_from(nodes)
    T.add_edges_from(edges)
    in_deg = T.in_degree()
    roots = [node for node in T if not in_deg[node]]
    if len(roots) != 1:
        raise Exception('expected one node with in-degree 1')
    root = roots[0]

    # Over-write the ordered list of edges, using a breadth first ordering.
    # Order the nodes according to the tail nodes of the ordered edges.
    edges = list(nx.bfs_edges(T, root))
    if len(edges) + 1 != len(nodes):
        raise Exception('expected the number of nodes to be one more '
                'than the number of edges')
    nodes = [root] + [tail for head, tail in edges]
    return nodes, edges


def flatten_for_rtaln(j_in):
    # flatten the ...
    pass


def unflatten_for_host():
    pass


def _test_bfs():
    ntrees = 10
    mu = 0.8
    for i in range(ntrees):
        nodes = [0]
        edges = []
        leaves = [0]
        while leaves:
            node = leaves.pop()
            nnew = np.random.poisson(mu)
            if nnew:
                latest = nodes[-1]
                new = list(range(latest+1, latest+1+nnew))
                nodes.extend(new)
                leaves.extend(new)
                edges.extend([[node, x] for x in new])

        print('original order:')
        print('nodes:', nodes)
        print('edges:', edges)
        print()

        if not edges:
            print('no edges...')
            continue

        # Rename the nodes arbitrarily but consistently,
        # and shuffle the edges.
        nodemap = list(nodes)
        random.shuffle(nodemap)
        nodes = [nodemap[n] for n in nodes]
        edges = [(nodemap[a], nodemap[b]) for a, b in edges]
        random.shuffle(edges)

        print('both nodes and edges are shuffled:')
        print('nodes:', nodes)
        print('edges:', edges)
        print()

        # Get the new orders.
        newnodes, newedges = reordered_nodes_and_edges(nodes, edges)

        print('reordered:')
        print('nodes:', newnodes)
        print('edges:', newedges)
        print()

        nodemap = dict((n, i) for i, n in enumerate(newnodes))
        nodes = [nodemap[n] for n in newnodes]
        edges = [(nodemap[a], nodemap[b]) for a, b in newedges]

        print('remapped:')
        print('nodes:', nodes)
        print('edges:', edges)
        print()


        # Check that after the tree is converted to a csr_matrix,
        # the indptr and the indices are as expected for the bf_tree ordering.
        row, col = zip(*edges)
        data = np.zeros(len(edges))
        m = csr_matrix((data, (row, col)))
        print('indptr:', m.indptr)
        print('indices:', m.indices)
        print()


if __name__ == '__main__':
    _test_bfs()
