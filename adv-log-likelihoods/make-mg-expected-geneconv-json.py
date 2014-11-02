"""
Make a JSON file to help compute expected gene conversions.

Copied from mle-geneconv-mle.py

"""
from __future__ import division, print_function, absolute_import

from itertools import izip_longest
import argparse
import json
import sys

import numpy as np

import mg94geneconv

from mg94geneconv import (
        MG94_GENECONV_Abstract,
        MG94_GENECONV_Concrete)

from mg_geneconv_common import (
        ad_hoc_fasta_reader,
        get_tree_info_with_outgroup)


# Official Python itertools recipe.
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def nts_to_codons(sequence):
    return [''.join(nts) for nts in grouper(sequence, 3)]


def main(args):
    # Read the hardcoded tree information.
    T, root = get_tree_info_with_outgroup()
    leaves = set(v for v, d in T.degree().items() if d == 1)
    outgroup = 'kluyveri'

    # Read the data as name sequence pairs.
    with open(args.fasta) as fin:
        name_seq_pairs = ad_hoc_fasta_reader(fin)

    # Convert from nucleotide sequences to codon sequences.
    name_seq_pairs = [
            (name, nts_to_codons(seq)) for name, seq in name_seq_pairs]

    # Throttle the number of sites if requested.
    if args.nsites is None:
        nsites = len(name_seq_pairs[0][1])
    else:
        nsites = args.nsites
        name_seq_pairs = [(n, s[:nsites]) for n, s in name_seq_pairs]
    if args.debug:
        print('number of sites to be analyzed:', nsites, file=sys.stderr)

    # Convert the pairs to a dict.
    name_to_seq = dict(name_seq_pairs)

    # Order the observables.
    suffix_len = 7
    observable_names = [name for name, seq in name_seq_pairs]
    observable_suffixes = [name[-suffix_len:] for name in observable_names]

    # For each observable, define the axis.
    # In our case, the compound state has one axis for each paralog,
    # and for each node zero or one or both paralogs may be observable.
    # Each observable can be observed only on a single axis.
    #suffix_to_axis = {'YAL056W' : 0, 'YOR371C' : 1}
    suffix_to_axis = {'YDR502C' : 0, 'YLR180W' : 1}
    observable_axes = [suffix_to_axis[s] for s in observable_suffixes]

    """
    # Define the map from nucleotide to observation index.
    nt_to_state = {
            'A' : 0,
            'C' : 1,
            'G' : 2,
            'T' : 3}
    """

    # Define the map from codon to observation index.
    codon_to_state = dict()
    for i, (codon, aa) in enumerate(mg94geneconv._gen_codon_aa_pairs()):
        codon_to_state[codon.upper()] = i

    iid_observations = []
    for site in range(nsites):
        observations = []
        for name in observable_names:
            observation = codon_to_state[name_to_seq[name][site]]
            observations.append(observation)
        iid_observations.append(observations)

    # Do something about the node representations.
    # They need to be numeric.
    # That is easy enough, I think they can be arbitrarily numbered.
    names = list(T)
    name_to_node = dict((n, i) for i, n in enumerate(names))
    edges = list(T.edges())
    edge_to_eidx = dict((e, i) for i, e in enumerate(edges))

    if args.debug:
        print(observable_names, file=sys.stderr)
        print(name_to_node, file=sys.stderr)
    observable_nodes = [name_to_node[n[:-suffix_len]] for n in observable_names]

    tree_row = [name_to_node[na] for na, nb in edges]
    tree_col = [name_to_node[nb] for na, nb in edges]
    tree_process = [0 if e == ('N0', outgroup) else 1 for e in edges]

    # define the process associated with an initial guess
    #M = MG94_GENECONV_Abstract()
    #guess = M.instantiate()
    #guess.set_kappa(2.0)
    #guess.set_omega(0.5)
    #guess.set_tau(2.0)
    #guess.set_nt_probs([0.30, 0.25, 0.20, 0.25])

    # define the initial edge rate guesses
    #edge_rates = [0.1] * len(edges)


    M = MG94_GENECONV_Abstract()
    guess = M.instantiate()

    guess.set_kappa(3.79231643078)
    guess.set_omega(0.0438633787983)
    guess.set_tau(0.574444741827)
    guess.set_nt_probs([0.26477356, 0.22461937, 0.20114206, 0.30946501])
    edge_rates = [
        #('N0', 'N1') : 
        0.203760528573,
        #('N0', 'kluyveri') :
        0.258685285947,
        #('N1', 'N2') : 
        0.2265435065,
        #('N1', 'castellii') :
        0.599417397961,
        #('N2', 'N3') :
        0.0858608166834,
        #('N2', 'bayanus') :
        0.131482447512,
        #('N3', 'N4') :
        0.0844600355092,
        #('N3', 'kudriavzevii') :
        0.171568821677,
        #('N4', 'N5') :
        0.0723182934745,
        #('N4', 'mikatae') :
        0.192393546629,
        #('N5', 'cerevisiae') :
        0.0918760080748,
        #('N5', 'paradoxus') :
        0.0897278941891,
        ]

    # create the initial unconstrained guess vector
    x_process = guess.get_x()
    x_rates = np.log(np.array(edge_rates))
    x = np.concatenate([x_process, x_rates])

    j_in = create_input_json_object(
            M,
            tree_row, tree_col, tree_process,
            observable_nodes, observable_axes, iid_observations,
            edges,
            x)

    print(json.dumps(j_in, indent=4))


def create_input_json_object(
        abstract_model,
        tree_row, tree_col, tree_process,
        observable_nodes, observable_axes, iid_observations,
        edges,
        x):
    # Deduce some counts.
    nsites = len(iid_observations)

    # All sites are weighted equally.
    site_weights = np.ones(nsites)

    # Break the opaque parameters into two pieces.
    # The first piece consists of parameters that affect the rate
    # matrix in complicated ways, and for which we will use finite-differences
    # to approximate sensitivities.
    # The second piece consists of edge-specific rate scaling factor
    # parameters whose sensitivities can be computed more efficiently
    k = len(edges)
    x_process, x_edge = x[:-k], x[-k:]

    tree = dict(
            row = tree_row,
            col = tree_col,
            rate = np.exp(x_edge).tolist(),
            process = tree_process,
            )

    # create the processes
    m0 = abstract_model.instantiate(x_process)
    m1 = abstract_model.instantiate(x_process)
    m0.set_tau(0)

    # define the pair of processes
    processes = []
    for m in m0, m1:
        row, col, rate, expect = m.get_sparse_rates_and_junk()
        p = dict(row=row, col=col, rate=rate, expect=expect)
        processes.append(p)

    # define the prior distribution
    prior_info = m0.get_distribution_info()
    prior_feasible_states, prior_distribution = prior_info

    # Build the nested structure to be converted to json.
    data = dict(
            node_count = len(edges) + 1,
            process_count = len(processes),
            state_space_shape = abstract_model.get_state_space_shape(),
            tree = tree,
            processes = processes,
            prior_feasible_states = prior_feasible_states,
            prior_distribution = prior_distribution.tolist(),
            observable_nodes = observable_nodes,
            observable_axes = observable_axes,
            iid_observations = iid_observations,
            )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--nsites', type=int,
            help='upper limit on the number of sites to be used')
    parser.add_argument('--fasta', required=True,
            help='fasta file with paralog alignment')
    main(parser.parse_args())
