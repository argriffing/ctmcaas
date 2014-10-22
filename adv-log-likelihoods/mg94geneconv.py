"""
Muse-Gaut 1994 continuous-time Markov model of codon evolution.

The gene conversion parameter tau is added.
The rate matrix will be scaled so that the expected number of codon changes
on a single paralog on a branch with rate scaling factor 1 will be 1,
if the gene conversion tau parameter is zero.

"""
from StringIO import StringIO
from itertools import product, permutations

import numpy as np
from numpy.testing import assert_equal


from npctmctree.models.base import AbstractModel, ConcreteModel
from npctmctree.models import genetic

__all__ = ['MG94_GENECONV_Abstract', 'MG94_GENECONV_Concrete']


def _gen_site_changes(sa, sb):
    for a, b in zip(sa, sb):
        if a != b:
            yield a, b


def _gen_codon_aa_pairs():
    nts = 'acgt'
    nt_to_idx = dict((nt, i) for i, nt in enumerate(nts))
    resids = []
    codons = []
    for line in StringIO(genetic.code).readlines():
        si, resid, codon = line.strip().split()
        if resid != 'stop':
            yield codon, resid


# Instances are not associated with actual parameter values.
class MG94_GENECONV_Abstract(AbstractModel):
    def __init__(self):
        self.structural_info = list(_get_structural_info())

    def get_state_space_shape(self):
        return (61, 61)

    def get_state_space_size(self):
        return np.prod(self.get_state_space_shape())

    def instantiate(self, x=None):
        return MG94_GENECONV_Concrete(self.structural_info, x)


# instances are associated with actual parameter values
class MG94_GENECONV_Concrete(ConcreteModel):
    def __init__(self, structural_info, x=None):
        """
        It is important that x can be an unconstrained vector
        that the caller does not need to know or care about.

        """
        # Unpack the parameters or use default values.
        if x is None:
            self.nt_probs = np.ones(4) / 4
            self.kappa = 2.0
            self.omega = 0.2
            self.tau = 1.2
            self.penalty = 0
        else:
            info = self._unpack_params(x)
            self.nt_probs, self.kappa, self.omega, self.tau, self.penalty = info

        # Get the structural info.
        self.structural_info = structural_info

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

    def set_omega(self, omega):
        self.omega = omega
        self._invalidate()

    def set_tau(self, tau):
        self.tau = tau
        self._invalidate()

    def set_nt_probs(self, nt_probs):
        self.nt_probs = nt_probs
        self._invalidate()

    def get_x(self):
        if self.x is None:
            self.x = self._pack_params(
                    self.nt_probs, self.kappa, self.omega, self.tau)
        return self.x

    def _pack_params(self, nt_distn1d, kappa, omega, tau):
        # helper function
        # This differs from the module-level function by not caring
        # about edge specific parameters.
        params = np.concatenate([nt_distn1d, [kappa, omega, tau]])
        log_params = np.log(params)
        return log_params

    def _unpack_params(self, log_params):
        # helper function
        # This differs from the module-level function by not caring
        # about edge specific parameters, and it does not create
        # the rate matrix.
        assert_equal(len(log_params.shape), 1)
        params = np.exp(log_params)
        nt_distn1d = params[:4]
        penalty = np.square(np.log(nt_distn1d.sum()))
        nt_distn1d = nt_distn1d / nt_distn1d.sum()
        kappa, omega, tau = params[-3:]
        return nt_distn1d, kappa, omega, tau, penalty

    def _process_sparse(self):
        distn_info, rate_triples = _get_distn_and_triples(
                self.structural_info,
                self.kappa, self.omega, self.tau, self.nt_probs)
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


def _get_distn_and_triples(structural_info, kappa, omega, tau, nt_probs):
    """
    Distribution and triples for the mg94 gene conversion process.

    """
    # Get the distribution for the single paralog MG94.
    # Also get the rate scaling factor.
    distn, triples, expected_rate = _get_mg94_info(kappa, omega, nt_probs)

    # Get the (head_state, tail_state, rate) triples,
    # where the head and tail states are decomposed according to sub-states.
    rate_triples = []
    for info in structural_info:
        sa, sb, ts, tv, nt, conversion, syn, nonsyn = info
        if ts or tv:
            mut_rate = (ts * kappa + tv) * nt_probs[nt]
        else:
            mut_rate = 0
        conv_rate = conversion * tau
        selection = syn + nonsyn * omega
        rate = (mut_rate / expected_rate + conv_rate) * selection
        triple = sa, sb, rate
        rate_triples.append(triple)

    # FIXME debugging...
    #print('found', len(rate_triples), 'structural triples')
    #print('expected something like', 61*61*((3+3+3+1) * 2))
    #print()

    # Get the sparse representation of the initial distribution.
    ncodons = len(distn)
    assert_equal(ncodons, 61)
    feasible_states = [(i, i) for i in range(ncodons)]
    distn_info = feasible_states, distn

    # Return information about the distribution and the rate triples.
    return distn_info, rate_triples


def _gen_plausible_transitions():
    """
    Each transition changes the state at exactly one axis.

    Some of these plausible transitions are not actually allowed under
    the codon gene conversion model.

    """
    ncodons = 61
    naxes = 2
    for sa in product(range(ncodons), repeat=naxes):
        sa0, sa1 = sa
        for sbk in range(ncodons):
            if sbk != sa0:
                yield (sa0, sa1), (sbk, sa1)
            if sbk != sa1:
                yield (sa0, sa1), (sa0, sbk)


def _get_dense_codon_structural_info():
    codons, resids = zip(*list(_gen_codon_aa_pairs()))
    ncodons = len(codons)
    naxes = 2
    assert_equal(ncodons, 61)
    nts = 'acgt'
    nt_to_idx = dict((nt, i) for i, nt in enumerate(nts))
    nt_transitions = {'ag', 'ga', 'ct', 'tc'}

    # Precompute codon state substitution properties.
    # These will be reused for the dicodon model.
    # (ts, tv, nt, syn, nonsyn)
    for sa, sb in permutations(range(ncodons), 2):

        # Determine the synonymous/nonsynonymous nature of the substitution.
        # The purpose of the redundancy of this notation is to facilitate
        # arithmetic with indicator variables.
        syn = (resids[sa] == resids[sb])
        nonsyn = not syn

        # Determine whether the change could arise by a nucleotide mutation.
        # If so, determine whether it is a transition or transversion,
        # and record the nucleotide index of the tail state.
        ts = False
        tv = False
        ca = codons[sa]
        cb = codons[sb]
        nt_changes = list(_gen_site_changes(ca, cb))
        if len(nt_changes) == 1:
            nta, ntb = nt_changes[0]
            tail_nt_state = nt_to_idx[ntb]
            if nta + ntb in nt_transitions:
                ts = True
            else:
                tv = True
        else:
            tail_nt_state = None

        yield sa, sb, ts, tv, tail_nt_state, syn, nonsyn



def _get_structural_info():
    """
    Yields tuples describing sparse rates due to mutation or gene conversion.
     * head state (a codon index pair)
     * tail state (a codon index pair)
     * transition {False, True}
     * transversion {False, True}
     * codon conversion {False, True}
     * synonymous {False, True}
     * nonsynonymous {False, True}
     * nucleotide tail state {None, 0, 1, 2, 3}

    """
    # Precompute codon state substitution properties.
    # These will be reused for the dicodon model.
    # (ts, tv, nt, syn, nonsyn)
    codon_index_pair_to_info = {}
    for info in _get_dense_codon_structural_info():
        sa, sb, ts, tv, tail_nt_state, syn, nonsyn = info
        codon_index_pair_to_info[sa, sb] = ts, tv, tail_nt_state, syn, nonsyn

    # Iterate over dicodon substitutions.
    for sa, sb in _gen_plausible_transitions():

        # Unpack states.
        sa0, sa1 = sa
        sb0, sb1 = sb

        # Assert that a substitution occurs on exactly one axis.
        codon_changes = list(_gen_site_changes(sa, sb))
        if len(codon_changes) != 1:
            raise Exception

        # Determine whether the change could arise from gene conversion.
        conversion = (sb0 == sb1)

        # Unpack the indices of the initial and final codon in the substitution.
        sak, sbk = codon_changes[0]

        # Look up the properties of the codon substitution.
        info = codon_index_pair_to_info[sak, sbk]
        ts, tv, tail_nt_state, syn, nonsyn = info

        # If either gene conversion or nucleotide mutation is possible,
        # then yield the information about the possible substitution.
        if ts or tv or conversion:
            yield sa, sb, ts, tv, tail_nt_state, conversion, syn, nonsyn


# This was the mg94 triples generator but has been slightly modified
# to return the unnormalized triples together with their normalization factor.
# This decomposition is more useful for the gene conversion model.
def _get_mg94_info(kappa, omega, nt_probs):
    """
    Parameters
    ----------
    kappa : float
        transition/transversion rate ratio
    omega : float
        synonymous/nonsynonymous rate ratio
    nt_probs : sequence of floats
        sequence of four probabilities in acgt order

    Returns
    -------
    distribution : sequence of floats
        stationary distribution
    triples : sequence of triples
        each triple is a (row, col, rate) triple
    expected_rate : float
        the expected rate

    """
    nts = 'acgt'
    nt_to_idx = dict((nt, i) for i, nt in enumerate(nts))
    resids = []
    codons = []
    for line in StringIO(genetic.code).readlines():
        si, resid, codon = line.strip().split()
        if resid != 'stop':
            resids.append(resid)
            codons.append(codon)
    ncodons = len(codons)
    assert_equal(ncodons, 61)

    distribution = []
    for codon in codons:
        probs = [nt_probs[nt_to_idx[nt]] for nt in codon]
        distribution.append(np.prod(probs))

    triples = []
    nt_transitions = {'ag', 'ga', 'ct', 'tc'}
    for i in range(ncodons):
        ri, ci = resids[i], codons[i]
        for j in range(ncodons):
            rj, cj = resids[j], codons[j]
            pairs = list(_gen_site_changes(ci, cj))
            if len(pairs) == 1:
                ni, nj = pairs[0]
                rate = np.prod([
                    nt_probs[nt_to_idx[nj]],
                    kappa if ni+nj in nt_transitions else 1,
                    omega if ri != rj else 1,
                    ])
                triples.append((i, j, rate))

    # scale the distribution and the rates to be friendly
    distribution = np.array(distribution) / sum(distribution)
    expected_rate = sum(distribution[i]*rate for i, j, rate in triples)

    # return the distribution and triples
    return distribution, triples, expected_rate
