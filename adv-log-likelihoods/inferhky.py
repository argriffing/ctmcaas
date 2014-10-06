"""
"""
from __future__ import division, print_function, absolute_import

import itertools
import argparse
import json

import numpy as np
import networkx as nx
from numpy.testing import assert_equal
from scipy.sparse import coo_matrix

from util import ad_hoc_fasta_reader


# Defines the shape of the state space,
# possibly does some precomputation,
# and links to the concrete class.
class HKY85_GENECONV_Abstract(object):
    def __init__(self):
        pass

    def get_state_space_shape(self):
        return (4, 4)

    def get_state_space_size(self):
        return np.prod(self.get_state_space_shape())

    def instantiate(self, x=None):
        return HKY85_GENECONV_Concrete(x)


class HKY85_GENECONV_Concrete(object):
    def __init__(self, x=None):
        """
        It is important that x can be an unconstrained vector
        that the caller does not need to know or care about.

        """
        # Unpack the parameters or use default values.
        if x is None:
            self.nt_probs = np.ones(4) / 4
            self.kappa = 2.0
            self.tau = 0.5
            self.penalty = 0
        else:
            info = self._unpack_params(x)
            self.nt_probs, self.kappa, self.tau, self.penalty = info

        # Mark some downstream attributes as not initialized.
        self._invalidate()

    def _invalidate(self):
        # this is called internally when parameter values change
        self.x = None
        self.prior_feasible_states = None
        self.prior_distribution = None
        self.row = None
        self.col = None
        self.rate = None

    def set_kappa(self, kappa):
        self.kappa = kappa
        self._invalidate()

    def set_tau(self, tau):
        self.tau = tau
        self._invalidate()

    def set_nt_probs(self, nt_probs):
        self.nt_probs = nt_probs
        self._invalidate()

    def get_x(self):
        if self.x is None:
            self.x = self._pack_params(self.nt_probs, self.kappa, self.tau)
        return self.x

    def _pack_params(self, nt_distn1d, kappa, tau):
        params = np.concatenate([nt_distn1d, [kappa, tau]])
        log_params = np.log(params)
        return log_params

    def _unpack_params(self, log_params):
        assert_equal(len(log_params.shape), 1)
        params = np.exp(log_params)
        nt_distn1d = params[:4]
        penalty = np.square(np.log(nt_distn1d.sum()))
        nt_distn1d = nt_distn1d / nt_distn1d.sum()
        kappa, tau = params[-2:]
        return nt_distn1d, kappa, tau, penalty

    def get_canonicalization_penalty(self):
        return self.penalty

    def _process_sparse(self):
        distn_info, rate_triples = _get_distn_and_triples(
                self.kappa, self.tau, self.nt_probs)
        self.prior_feasible_states, self.prior_distribution = distn_info
        self.row, self.col, self.rate = zip(*rate_triples)

    def get_distribution_info(self):
        # return a list of feasible states
        # and a list of corresponding probabilities
        if self.prior_feasible_states is None:
            self._process_sparse()
        return self.prior_feasible_states, self.prior_distribution

    def get_sparse_rates(self):
        # return information required to create the rate matrix
        if self.prior_feasible_states is None:
            self._process_sparse()
        return self.row, self.col, self.rate


def _gen_site_changes(ca, cb):
    for a, b in zip(ca, cb):
        if a != b:
            yield a, b


def _get_distn_and_triples(kappa, tau, nt_probs):
    """
    Parameters
    ----------
    kappa : float
        transition/transversion rate ratio
    tau : float
        gene conversion rate parameter
    nt_probs : sequence of floats
        sequence of four probabilities in acgt order

    Returns
    -------
    distribution_info : (feasible_states, distribution)
        sparse representation of distribution over initial states
    triples : sequence of triples
        Each element of the sequence is like (row, col, rate)
        where each element of row and of col is a pair.

    """
    nts = 'acgt'

    # Make a dense hky matrix with the right scaling.
    # TODO re-use the hky model in npctmctree.models
    distribution = nt_probs
    triples = []
    nt_transitions = {'ag', 'ga', 'ct', 'tc'}
    for i, ni in enumerate(nts):
        for j, nj in enumerate(nts):
            if i != j:
                rate = np.prod([
                    nt_probs[j],
                    kappa if ni+nj in nt_transitions else 1,
                    ])
                triples.append((i, j, rate))
    distribution = np.array(distribution) / sum(distribution)
    expected_rate = sum(distribution[i]*rate for i, j, rate in triples)
    triples = [(i, j, rate/expected_rate) for i, j, rate in triples]
    row, col, rate = zip(*triples)
    Q_hky = coo_matrix((rate, (row, col)), (4, 4)).A

    # Make the triples for the rate matrix with the larger state space.
    triples = []

    # If the hamming distance of the compound states is 1,
    # then the substitution is allowed, and its rate is at least
    # as large as the corresponding hky substitution rate.
    # If in addition the final compound state has the same nucleotide
    # in both paralogs, the gene conversion rate is added.
    states = list(itertools.product(range(4), repeat=2))
    for sa in states:
        for sb in states:
            site_changes = list(_gen_site_changes(sa, sb))
            if len(site_changes) == 1:
                head, tail = site_changes[0]
                if sb[0] == sb[1]:
                    rate = Q_hky[head, tail] + tau
                else:
                    rate = Q_hky[head, tail]
                triples.append((sa, sb, rate))

    # Force the prior to be pre-duplication,
    # so both paralogs have the same state,
    # and the distribution of this state is controlled by the hky model.
    prior_feasible_states = [[0, 0], [1, 1], [2, 2], [3, 3]]
    prior_distribution = nt_probs
    distn_info = prior_feasible_states, prior_distribution

    # Return the distribution and triples.
    return distn_info, triples


def get_tree_info_with_outgroup():
    T = nx.DiGraph()
    root = 'N0'
    T.add_edges_from([
            ('N0', 'Tamarin'),
            ('N0', 'N1'),
            ('N1', 'Macaque'),
            ('N1', 'N2'),
            ('N2', 'Orangutan'),
            ('N2', 'N3'),
            ('N3', 'Chimpanzee'),
            ('N3', 'Gorilla'),
            ])
    return T, root


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
    nt_probs = np.array([ 0.28289892,  0.25527043,  0.20734073,  0.25448992])
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

    tree = dict(
            row = [name_to_node[na] for na, nb in edges],
            col = [name_to_node[nb] for na, nb in edges],
            rate = [edge_to_rate[e] for e in edges],
            process = [0 if e == ('N0', 'Tamarin') else 1 for e in edges],
            )

    M = HKY85_GENECONV_Abstract()

    # create a process
    m1 = M.instantiate()
    m1.set_kappa(kappa)
    m1.set_tau(tau)
    m1.set_nt_probs(nt_probs)

    # create the other process
    m0 = M.instantiate(m1.get_x())
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
            node_count = len(T),
            process_count = 2,
            state_space_shape = (4, 4),
            tree = tree,
            processes = processes,
            prior_feasible_states = prior_feasible_states,
            prior_distribution = prior_distribution.tolist(),
            observable_nodes = observable_nodes,
            observable_axes = observable_axes,
            iid_observations = iid_observations,
            )

    print(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True,
            help='fasta file with paralog alignment of EDN and ECP')
    main(parser.parse_args())

