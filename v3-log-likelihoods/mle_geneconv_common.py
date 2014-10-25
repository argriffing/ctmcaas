"""
Common functions for gene conversion MLE.

Copied and pasted from ctmcaas/adv-likelihood/mle_geneconv_common.py.

Some code related to command-line and distributed execution has been removed.

"""
from __future__ import division, print_function

import functools
import sys
import json

import numpy as np
from numpy.testing import assert_equal


__all__ = ['objective_and_gradient']


def objective_and_gradient(
        abstract_model,
        json_evaluation_function,
        tree_row, tree_col, tree_process,
        observable_nodes, observable_axes, iid_observations,
        edges,
        x):
    """
    The x argument is the opaque 1d vector of parameters.

    This requires an evaluator that knows about the derivative
    of the log likelihood with respect to parameter values.

    Hard-code the delta for non-edge-rate finite differences.
    The intention is to use the default value used in L-BFGS-B
    in scipy.optimize.minimize.

    """
    delta = 1e-8

    # Break the opaque parameters into two pieces.
    # The first piece consists of parameters that affect the rate
    # matrix in complicated ways, and for which we will use finite-differences
    # to approximate sensitivities.
    # The second piece consists of edge-specific rate scaling factor
    # parameters whose sensitivities can be computed more efficiently
    k = len(edges)
    x_process, x_edge = x[:-k], x[-k:]

    # For the first call, request derivatives for all edges.
    requested_derivatives = list(range(k))
    ll, edge_derivs = _log_likelihood_and_edge_derivatives(
            requested_derivatives,
            abstract_model,
            json_evaluation_function,
            tree_row, tree_col, tree_process,
            observable_nodes, observable_axes, iid_observations,
            edges,
            x)

    # Count the number of parameters that are not
    # edge-specific rate scaling factors.
    m = len(x) - k

    # For subsequent calls, use finite differences to estimate
    # derivatives with respect to these parameters.
    other_derivs = []
    requested_derivatives = []
    for i in range(m):
        x_plus_delta = np.array(x)
        x_plus_delta[i] += delta
        ll_delta, _ = _log_likelihood_and_edge_derivatives(
                requested_derivatives,
                abstract_model,
                json_evaluation_function,
                tree_row, tree_col, tree_process,
                observable_nodes, observable_axes, iid_observations,
                edges,
                x_plus_delta)
        d_estimate = (ll_delta - ll) / delta
        other_derivs.append(d_estimate)
    other_derivs = np.array(other_derivs)

    print('other derivatives:', other_derivs, file=sys.stderr)

    # Return the function value and the gradient.
    # Remember this is to be minimized so convert this to use signs correctly.
    f = -ll
    g = -np.concatenate((other_derivs, edge_derivs))
    print('objective function:', f, file=sys.stderr)
    print('gradient:', g, file=sys.stderr)
    return f, g


def _log_likelihood_and_edge_derivatives(
        requested_derivatives,
        abstract_model,
        json_evaluation_function,
        tree_row, tree_col, tree_process,
        observable_nodes, observable_axes, iid_observations,
        edges,
        x):
    """
    Evaluate the log likelihood and some of its derivatives.
    
    The evaluated derivatives are the ones that correspond
    to edge-specific scaling factor parameters.

    Parameters
    ----------
    requested_derivatives : sequence of edge indices
        Indicates for which edges to compute log likelihood derivatives.
        For each edge index in this list,
        the derivative of the log likelihood with respect to
        the logarithm of the edge-specific rate scaling factor
        will be computed.
    abstract_model : object
        This argument has the information about the model.
        It knows about the sparsity structure of the rate matrix,
        and for an opaque vector of unconstrained transformed parameter values
        it can return a concrete instantiation of that model
        with the suitably transformed parameter values
        from which the actual rates of the matrix can be extracted.
    json_evaluation_function : callable
        A function like jsonctmctree.ll.process_json_in

    """
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
        row, col, rate = m.get_sparse_rates()
        p = dict(row=row, col=col, rate=rate)
        processes.append(p)

    # define the prior distribution
    prior_info = m0.get_distribution_info()
    prior_feasible_states, prior_distribution = prior_info

    # Build the nested structure to be converted to json.
    j_in = dict(
            site_weights = site_weights.tolist(),
            requested_derivatives = requested_derivatives,
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

    # FIXME debug...
    with open('out.json', 'wt') as fout:
        s = json.dumps(j_in, indent=4)
        print(s, file=fout)

    j_out = json_evaluation_function(j_in)

    status = j_out['status']
    feasibility = j_out['feasibility']

    if status != 'success' or not feasibility:
        print('results:', file=sys.stderr)
        print(j_out, file=sys.stderr)
        raise Exception('encountered some problem in the calculation of '
                'log likelihood and its derivatives')

    log_likelihood = j_out['log_likelihood']
    edge_derivatives = j_out['edge_derivatives']

    print('log likelihood:', log_likelihood, file=sys.stderr)
    print('edge derivatives:', edge_derivatives, file=sys.stderr)

    return log_likelihood, edge_derivatives
