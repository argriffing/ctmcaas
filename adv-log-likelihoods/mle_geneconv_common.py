"""
Common functions for gene conversion MLE.

"""
import json
import subprocess
import requests

import numpy as np

import ll


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
