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

from hky_geneconv_common import (
        ad_hoc_fasta_reader,
        get_tree_info_with_outgroup,
        HKY85_GENECONV_Abstract,
        HKY85_GENECONV_Concrete)


def eval_ll_cmdline(j_data):
    ll_input_string = json.dumps(j_data)
    p = subprocess.Popen(
            ['python', 'll.py'],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE)
    outdata, errdata = p.communicate(input=ll_input_string)
    j_ll = json.loads(outdata)
    return j_ll


def eval_ll_internets(url, j_data):
    return requests.post(url, data=json.dumps(j_data)).json()


def objective(
        fn,
        tree_row, tree_col, tree_process,
        observable_nodes, observable_axes, iid_observations,
        edges,
        x):
    """
    The x argument is the opaque 1d vector of parameters.

    """
    # break the opaque parameters into two pieces
    k = len(edges)
    x_process, x_edge = x[:-k], x[-k:]

    tree = dict(
            row = tree_row,
            col = tree_col,
            rate = np.exp(x_edge).tolist(),
            process = tree_process,
            )

    # create the processes
    M = HKY85_GENECONV_Abstract()
    m0 = M.instantiate(x_process)
    m1 = M.instantiate(x_process)
    m0.set_tau(0)

    # define the pair of processes
    processes = []
    for m in m0, m1:
        row, col, rate = m.get_sparse_rates()
        p = dict(row=row, col=col, rate=rate)
        processes.append(p)

    # define the prior distribution
    prior_info = m0.get_distribution_info()
    prior_feasible_states, prior_distribution = prior_info

    # Build the nested structure to be converted to json.
    data = dict(
            node_count = len(edges) + 1,
            process_count = len(processes),
            state_space_shape = M.get_state_space_shape(),
            tree = tree,
            processes = processes,
            prior_feasible_states = prior_feasible_states,
            prior_distribution = prior_distribution.tolist(),
            observable_nodes = observable_nodes,
            observable_axes = observable_axes,
            iid_observations = iid_observations,
            )

    j_ll = fn(data)

    log_likelihood = sum(j_ll['log_likelihoods'])
    y = -log_likelihood
    print('value of objective:', y)
    return y


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

    # Use previously computed max likelihood parameter estimates.
    # The log likelihood should be near 1721.7834201000449.
    kappa = 2.11379742986
    tau = 1.82001290642
    nt_probs = [ 0.28289892,  0.25527043,  0.20734073,  0.25448992]
    edge_rate_pairs = (
        (('N0', 'N1'), 0.0706039486132),
        (('N0', 'Tamarin'), 0.102976327457),
        (('N1', 'Macaque'), 0.0511418556427),
        (('N1', 'N2'), 0.00879371918394),
        (('N2', 'N3'), 0.0109200792917),
        (('N2', 'Orangutan'), 0.0298655988153),
        (('N3', 'Gorilla'), 0.00501349585464),
        (('N3', 'Chimpanzee'), 0.00455294684035),
        )
    edge_to_rate = dict(edge_rate_pairs)

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

    # define the source of the log likelihood evaluation
    if args.ll_url:
        fn = functools.partial(eval_ll_internets, args.ll_url)
    else:
        fn = eval_ll_cmdline

    # define the function to minimize
    f = functools.partial(
            objective,
            fn,
            tree_row, tree_col, tree_process,
            observable_nodes, observable_axes, iid_observations,
            edges)

    # do the search
    result = scipy.optimize.minimize(f, x, method='L-BFGS-B')

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
