from __future__ import division, print_function, absolute_import

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

from hky85geneconv import (
        HKY85_GENECONV_Abstract,
        HKY85_GENECONV_Concrete)

from hky_geneconv_common import (
        ad_hoc_fasta_reader,
        get_tree_info_with_outgroup)


def main(args):
    # Read the hardcoded tree information.
    T, root = get_tree_info_with_outgroup()
    leaves = set(v for v, d in T.degree().items() if d == 1)
    outgroup = 'Tamarin'

    # Read the data as name sequence pairs.
    with open(args.fasta) as fin:
        name_seq_pairs = ad_hoc_fasta_reader(fin)
    name_to_seq = dict(name_seq_pairs)

    # Order the observables.
    observable_names = [name for name, seq in name_seq_pairs]
    observable_suffixes = [name[-3:] for name in observable_names]

    # For each observable, define the axis.
    # In our case, the compound state has one axis for each paralog,
    # and for each node zero or one or both paralogs may be observable.
    # Each observable can be observed only on a single axis.
    suffix_to_axis = {'EDN' : 0, 'ECP' : 1}
    observable_axes = [suffix_to_axis[s] for s in observable_suffixes]

    # Define the map from nucleotide to observation index.
    nt_to_state = {
            'A' : 0,
            'C' : 1,
            'G' : 2,
            'T' : 3}

    # Track the state observations.
    nsites = len(name_seq_pairs[0][1])
    iid_observations = []
    for site in range(nsites):
        observations = []
        for name in observable_names:
            observation = nt_to_state[name_to_seq[name][site]]
            observations.append(observation)
        iid_observations.append(observations)

    # Do something about the node representations.
    # They need to be numeric.
    # That is easy enough, I think they can be arbitrarily numbered.
    names = list(T)
    name_to_node = dict((n, i) for i, n in enumerate(names))
    edges = list(T.edges())
    edge_to_eidx = dict((e, i) for i, e in enumerate(edges))

    observable_nodes = [name_to_node[n[:-3]] for n in observable_names]

    tree_row = [name_to_node[na] for na, nb in edges]
    tree_col = [name_to_node[nb] for na, nb in edges]
    tree_process = [0 if e == ('N0', 'Tamarin') else 1 for e in edges]

    # define the process associated with an initial guess
    M = HKY85_GENECONV_Abstract()
    guess = M.instantiate()
    guess.set_kappa(2.0)
    guess.set_tau(2.0)
    guess.set_nt_probs([0.30, 0.25, 0.20, 0.25])

    # define the initial edge rate guesses
    edge_rates = [0.1] * len(edges)

    # create the initial unconstrained guess vector
    x_process = guess.get_x()
    x_rates = np.log(np.array(edge_rates))
    x = np.concatenate([x_process, x_rates])

    # debugging
    USE_FINITE_DIFFERENCES = False
    if USE_FINITE_DIFFERENCES:
        fn_source = mle_geneconv_common.eval_ll_module
        fn_strategy = mle_geneconv_common.objective_and_finite_differences
        jac = True
    else:
        #nworkers = 4
        #fn_source = functools.partial(
                #mle_geneconv_common.eval_ll_v3module_multiprocessing,
                #nworkers)
        fn_source = mle_geneconv_common.eval_ll_v3module
        fn_strategy = mle_geneconv_common.objective_and_gradient
        jac = True


    """
    # define the source of the log likelihood evaluation
    if args.ll_url:
        fn = functools.partial(
                mle_geneconv_common.eval_ll_internets,
                args.ll_url)
    else:
        #fn = mle_geneconv_common.eval_ll_cmdline
        fn = mle_geneconv_common.eval_ll_module
        #fn = mle_geneconv_common.eval_ll_v3module
    """

    # define the abstract model
    M = HKY85_GENECONV_Abstract()

    # define the function to minimize
    f = functools.partial(
            fn_strategy,
            M,
            fn_source,
            tree_row, tree_col, tree_process,
            observable_nodes, observable_axes, iid_observations,
            edges)

    # do the search
    result = scipy.optimize.minimize(f, x, jac=jac, method='L-BFGS-B')

    # report the raw search results
    print('optimization result:')
    print(result)
    print()

    # unpack the result
    k = len(edges)
    x = result.x
    x_process, x_edge = x[:-k], x[-k:]
    edge_rates = np.exp(x_edge)
    m = M.instantiate(x_process)
    print('kappa:', m.kappa)
    print('tau:', m.tau)
    print('nt_probs:', m.nt_probs)
    print('edge rates:')
    for edge, rate in zip(edges, edge_rates):
        print(edge, ':', rate)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_url')
    parser.add_argument('--fasta', required=True,
            help='fasta file with paralog alignment of EDN and ECP')
    main(parser.parse_args())
