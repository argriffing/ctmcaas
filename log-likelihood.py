"""
A wrapper for log likelihoods for ctmcs on trees.

Read a json file on stdin and write one on stdout.

{
	"tree" :
	{
		"order" : int,
		"row" : [int, ...],
		"col" : [int, ...],
		"data" : [float, ...]
	}
	"rates" :
	{
		"order" : int,
		"row" : [int, ...],
		"col" : [int, ...],
		"data" : [float, ...]
	}
	"prior" : [float, ...],
	"observable_nodes" : [int, ...],
	"iid_observations" : [
		[int, ...],
		[int, ...],
		...
		[int, ...]]
}


{
	"status" : <"success" OR "error">,
	"message" : "string",
	"feasibilities" : [int, ...],
	"log_likelihoods" : [float, ...]
}

"""
from __future__ import print_function, division

import argparse
import sys
import json
import traceback

import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.sparse import coo_matrix

from npmctree import dynamic_xmap_lhood

def get_tree_info(j_in):
    tree = j_in['tree']
    nnodes = tree['order']
    nodes = set(range(nnodes))
    row = tree['row']
    col = tree['col']
    data = np.array(tree['data'], dtype=float)
    if not (set(row) <= nodes):
        raise Exception('unexpected node')
    if not (set(col) <= nodes):
        raise Exception('unexpected node')
    T = nx.DiGraph()
    T.add_nodes_from(range(nnodes))
    T.add_edges_from(zip(row, col))
    if len(T.edges()) + 1 != len(T):
        raise Exception('expected the number of edges to be one more '
                'than the number of nodes')
    in_degree = T.in_degree()
    roots = [n for n in nodes if in_degree[n] == 0]
    if len(roots) != 1:
        raise Exception('expected exactly one root')
    for i in range(nnodes):
        T.in_degree()
    root = roots[0]
    edge_rate_pairs = [((r, c), d) for r, c, d in zip(row, col, data)]
    return T, root, edge_rate_pairs


def get_rates_info(j_in):
    rates = j_in['rates']
    nstates = rates['order']
    row = rates['row']
    col = rates['col']
    data = np.array(rates['data'], dtype=float)
    distn = np.array(j_in['prior'], dtype=float)
    Q = coo_matrix((data, (row, col)), shape=(nstates, nstates)).A
    exit_rates = Q.sum(axis=1)
    Q = Q - np.diag(exit_rates)
    return Q, np.array(distn)


def process_json_in(j_in):

    # read the tree
    T, root, edge_rate_pairs = get_tree_info(j_in)

    # read the rate matrix and the prior distribution at the root
    Q, distn = get_rates_info(j_in)

    # create the input to the function that computes i.i.d. log likelihoods
    edge_to_P = dict((e, expm(r*Q)) for e, r in edge_rate_pairs)

    # get information related to observations
    observable_nodes = j_in['observable_nodes']
    iid_observations = j_in['iid_observations']
    xmaps = [dict(zip(observable_nodes, obs)) for obs in iid_observations]

    # get the log likelihoods
    likelihoods = dynamic_xmap_lhood.get_iid_lhoods(
            T, edge_to_P, root, distn, xmaps)
    log_likelihoods = np.log(likelihoods)

    # adjust for infeasibility
    feasibilities = np.isfinite(log_likelihoods)
    log_likelihoods = np.where(feasibilities, log_likelihoods, 0)

    # create the output in a format that json will like
    j_out = dict(
            status = 'success',
            feasibilities = feasibilities.astype(int).tolist(),
            log_likelihoods = log_likelihoods.tolist())

    return j_out


def main(args):
    try:
        s_in = sys.stdin.read()
        print('input:')
        print(s_in)
        print()
        j_in = json.loads(s_in)
    except Exception as e:
        return dict(
                status = 'error',
                message = 'json parsing error: ' + traceback.format_exc())
    try:
        return process_json_in(j_in)
    except Exception as e:
        return dict(
                status = 'error',
                message = 'processing error: ' + traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    j_out = main(parser.parse_args())
    print(json.dumps(j_out))
