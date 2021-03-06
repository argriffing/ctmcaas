from __future__ import division, print_function, absolute_import

from itertools import izip_longest
import functools
import argparse
import json
import subprocess
import urllib
import urllib2
import requests

import numpy as np
import scipy.optimize

import mle_geneconv_common
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
    print('number of sites to be analyzed:', nsites)

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

    print(observable_names)
    print(name_to_node)
    observable_nodes = [name_to_node[n[:-suffix_len]] for n in observable_names]

    tree_row = [name_to_node[na] for na, nb in edges]
    tree_col = [name_to_node[nb] for na, nb in edges]
    tree_process = [0 if e == ('N0', outgroup) else 1 for e in edges]

    # define the process associated with an initial guess
    M = MG94_GENECONV_Abstract()
    guess = M.instantiate()
    guess.set_kappa(2.0)
    guess.set_omega(0.5)
    guess.set_tau(2.0)
    guess.set_nt_probs([0.30, 0.25, 0.20, 0.25])

    # define the initial edge rate guesses
    edge_rates = [0.1] * len(edges)

    # create the initial unconstrained guess vector
    x_process = guess.get_x()
    x_rates = np.log(np.array(edge_rates))
    x = np.concatenate([x_process, x_rates])

    # define the source of the log likelihood evaluation
    if args.ll_url:
        fn = functools.partial(
                mle_geneconv_common.eval_ll_internets,
                args.ll_url)
    else:
        #fn = mle_geneconv_common.eval_ll_cmdline
        #fn = mle_geneconv_common.eval_ll_module
        fn = mle_geneconv_common.eval_ll_v3module

        #nworkers = 4
        #fn = functools.partial(
                #mle_geneconv_common.eval_ll_v3module_multiprocessing,
                #nworkers)


    # define the function to minimize
    f = functools.partial(
            #mle_geneconv_common.objective,
            mle_geneconv_common.objective_and_gradient,
            M,
            fn,
            tree_row, tree_col, tree_process,
            observable_nodes, observable_axes, iid_observations,
            edges)

    # print intermediate results in the callback
    def cb(x):
        k = len(edges)
        x_process, x_edge = x[:-k], x[-k:]
        edge_rates = np.exp(x_edge)
        m = M.instantiate(x_process)
        print('kappa:', m.kappa)
        print('omega:', m.omega)
        print('tau:', m.tau)
        print('nt_probs:', m.nt_probs)
        print('edge rates:')
        for edge, rate in zip(edges, edge_rates):
            print(edge, ':', rate)
        print()

    # do the search
    result = scipy.optimize.minimize(
            f, x, jac=True, method='L-BFGS-B', callback=cb)

    # report the raw search results
    print('optimization result:')
    print(result)
    print()

    # unpack the result and print it
    x = result.x
    cb(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_url')
    parser.add_argument('--nsites', type=int,
            help='upper limit on the number of sites to be used')
    parser.add_argument('--fasta', required=True,
            help='fasta file with paralog alignment')
    main(parser.parse_args())
