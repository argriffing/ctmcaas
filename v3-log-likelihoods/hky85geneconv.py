"""
HKy85+geneconv with an interface friendly to the log likelihood calculator.

The alignment and the tree are not considered here.

"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
from numpy.testing import assert_equal
from scipy.sparse import coo_matrix

__all__ = ['HKY85_GENECONV_Abstract', 'HKY85_GENECONV_Concrete']


def _gen_site_changes(ca, cb):
    for a, b in zip(ca, cb):
        if a != b:
            yield a, b


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
        #distn_info, rate_triples = _get_distn_and_triples(
                #self.kappa, self.tau, self.nt_probs)
        #self.prior_feasible_states, self.prior_distribution = distn_info
        #self.row, self.col, self.rate = zip(*rate_triples)

        # FIXME
        # This has been modified to also compute gene conversion proportion.
        distn_info, rate_info = _get_distn_and_triples_ex(
                self.kappa, self.tau, self.nt_probs)
        self.prior_feasible_states, self.prior_distribution = distn_info
        self.row, self.col, self.rate, self.expect = zip(*rate_info)

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

    def get_sparse_rates_and_geneconv_proportion(self):
        # FIXME
        # This has been added for testing the calculation of expectations.
        #
        # Return information required to create the rate matrix
        if self.prior_feasible_states is None:
            self._process_sparse()
        return self.row, self.col, self.rate, self.expect


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


#FIXME
# This has been copied and pasted to add expectation information
# in an ad-hoc way.
def _get_distn_and_triples_ex(kappa, tau, nt_probs):
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
    info : sequence of tuples
        This used to be a sequence of (row, col, rate) triples,
        but now an extra term has been added for computing
        gene conversion expectations.

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
    info = []

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
                    gene_conversion_proportion = tau / rate
                else:
                    rate = Q_hky[head, tail]
                    gene_conversion_proportion = 0
                info.append((sa, sb, rate, gene_conversion_proportion))

    # Force the prior to be pre-duplication,
    # so both paralogs have the same state,
    # and the distribution of this state is controlled by the hky model.
    prior_feasible_states = [[0, 0], [1, 1], [2, 2], [3, 3]]
    prior_distribution = nt_probs
    distn_info = prior_feasible_states, prior_distribution

    # Return the distribution and info.
    return distn_info, info
