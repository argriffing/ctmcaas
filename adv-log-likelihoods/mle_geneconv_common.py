"""
Common functions for gene conversion MLE.

"""
import json
import subprocess
import requests

import numpy as np

import ll

import jsonctmctree.ll


__all__ = ['eval_ll_cmdline', 'eval_ll_internets', 'objective']


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


def eval_ll_module(j_data):
    return ll.process_json_in(j_data)


def eval_ll_v3module(j_data):
    return jsonctmctree.ll.process_json_in(j_data)


def _log_likelihood_and_edge_derivatives(
        requested_derivatives,
        abstract_model,
        fn,
        tree_row, tree_col, tree_process,
        observable_nodes, observable_axes, iid_observations,
        edges,
        x):
    """
    Evaluate the log likelihood and some of its derivatives.
    
    The evaluated derivatives are the ones that correspond
    to edge-specific scaling factor parameters.

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
    data = dict(
            site_weights = site_weights,
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

    j_ll = fn(data)

    log_likelihood = j_ll['log_likelihood']
    edge_derivatives = j_ll['edge_derivatives']

    print('log likelihood:', log_likelihood)
    print('edge derivatives:', edge_derivatives)

    return log_likelihood, edge_derivatives


def objective_and_gradient(
        abstract_model,
        fn,
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
            fn,
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
                fn,
                tree_row, tree_col, tree_process,
                observable_nodes, observable_axes, iid_observations,
                edges,
                x_plus_delta)
        d_estimate = (ll_delta - ll) / delta
        other_derivs.append(d_estimate)
    other_derivs = np.array(other_derivs)

    print('other derivatives:', other_derivs)

    # Return the function value and the gradient.
    # Remember this is to be minimized so convert this to use signs correctly.
    f = -ll
    g = -np.concatenate((other_derivs, edge_derivs))
    return f, g


def objective_and_finite_differences(
        abstract_model,
        fn,
        tree_row, tree_col, tree_process,
        observable_nodes, observable_axes, iid_observations,
        edges,
        x):
    """
    Use finite differences in the same way as the default L-BFGS-B.

    This function uses finite differences for all parameters,
    not just the ones that are not edge-specific rates.

    """
    delta = 1e-8
    curried_objective = functools.partial(
            objective,
            abstract_model,
            fn,
            tree_row, tree_col, tree_process,
            observable_nodes, observable_axes, iid_observations,
            edges)
    ll = curried_objective(x)
    n = len(x)
    diffs = []
    for i in range(n):
        u = np.array(x)
        u[i] += delta
        ll_diff = curried_objective(u)
        d = (ll_diff - ll) / delta
        diffs.append(d)
    #TODO still under construction and not working
    return np.array(diffs)


def objective(
        abstract_model,
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

    j_ll = fn(data)

    log_likelihood = sum(j_ll['log_likelihoods'])
    y = -log_likelihood
    print('value of objective:', y)
    return y
