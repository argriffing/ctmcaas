"""
Implement log likelihoods for complicated models.

This interface takes some care about memory usage,
while allowing more subtlety in the representation of observed data,
and while allowing more flexibility in the representation of
inhomogeneity of the process across branches.

"""
from __future__ import print_function, division

import argparse
import json
import traceback
import sys

import networkx as nx
import numpy as np
from numpy.testing import assert_equal
from scipy.linalg import expm, eig, inv
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import coo_matrix


def get_observables_info(j_in):
    return (
            np.array(j_in['observable_nodes']),
            np.array(j_in['observable_axes']),
            np.array(j_in['iid_observations']))


def get_prior_info(j_in):
    return (
            np.array(j_in['prior_feasible_states']),
            np.array(j_in['prior_distribution'], dtype=float))


def get_processes_info(j_in):
    processes_row = []
    processes_col = []
    processes_rate = []
    for j_process in j_in['processes']:
        processes_row.append(j_process['row'])
        processes_col.append(j_process['col'])
        processes_rate.append(j_process['rate'])
    return (
            np.array(processes_row),
            np.array(processes_col),
            np.array(processes_rate, dtype=float))


def get_tree_info(j_in):
    node_count = j_in['node_count']
    process_count = j_in['process_count']
    tree = j_in['tree']
    nodes = set(range(node_count))
    row = tree['row']
    col = tree['col']
    rate = np.array(tree['rate'], dtype=float)
    process = np.array(tree['process'])
    if not (set(row) <= nodes):
        raise Exception('unexpected node')
    if not (set(col) <= nodes):
        raise Exception('unexpected node')
    T = nx.DiGraph()
    T.add_nodes_from(range(node_count))
    edges = zip(row, col)
    T.add_edges_from(edges)
    if len(T.edges()) != len(edges):
        raise Exception('the tree has an unexpected number of edges')
    if len(edges) + 1 != len(T):
        raise Exception('expected the number of edges to be one more '
                'than the number of nodes')
    in_degree = T.in_degree()
    roots = [n for n in nodes if in_degree[n] == 0]
    if len(roots) != 1:
        raise Exception('expected exactly one root')
    for i in range(node_count):
        T.in_degree()
    root = roots[0]
    edges = zip(row, col)
    edge_rate_pairs = zip(edges, rate)
    edge_process_pairs = zip(edges, process)
    return T, root, edges, edge_rate_pairs, edge_process_pairs


def get_node_to_depth(T, root):
    node_to_depth = {root : 0}
    for head, tail in nx.bfs_edges(T, root):
        node_to_depth[tail] = node_to_depth[head] + 1
    return node_to_depth


def get_node_to_subtree_depth(T, root):
    subdepth = {}
    for node in reversed(list(nx.bfs_nodes(T, root))):
        successors = T.successors(tail)
        if not successors:
            subdepth[node] = 0
        else:
            subdepth[node] = max(subdepth[s] for s in successors) + 1
    return subdepth


def get_node_to_subtree_thickness(T, root):
    thickness = {}
    for node in nx.dfs_postorder_nodes(T, root):
        successors = T.successors(node)
        if not successors:
            thickness[node] = 1
        else:
            w = sorted((thickness[n] for n in successors), reverse=True)
            thickness[node] = max(w + i for i, w in enumerate(w))
    return thickness


def get_node_evaluation_order(T, root):
    thickness = get_node_to_subtree_thickness(T, root)
    expanded = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n in expanded:
            yield n
        else:
            successors = list(T.successors(n))
            if not successors:
                yield n
            else:
                expanded.add(n)
                pairs = [(thickness[x], x) for x in successors]
                progeny = [x for w, x in sorted(pairs)]
                stack.extend([n] + progeny)


def create_sparse_rate_matrix(state_space_shape, row, col, rate):
    """
    Create the rate matrix.

    """
    # check conformability of input arrays
    ndim = len(state_space_shape)
    assert_equal(len(row.shape), 2)
    assert_equal(len(col.shape), 2)
    assert_equal(len(rate.shape), 1)
    assert_equal(row.shape[0], rate.shape[0])
    assert_equal(col.shape[0], rate.shape[0])
    assert_equal(row.shape[1], ndim)
    assert_equal(col.shape[1], ndim)

    # create the sparse Q matrix from the sparse arrays
    nstates = np.prod(state_space_shape)
    mrow = np.ravel_multi_index(row.T, state_space_shape)
    mcol = np.ravel_multi_index(col.T, state_space_shape)
    Q = coo_matrix((rate, (mrow, mcol)), (nstates, nstates))

    # get the dense array of exit rates, and set the diagonal
    exit_rates = Q.sum(axis=1).A.flatten()
    Q.setdiag(-exit_rates)

    return Q


def create_dense_rate_matrix(state_space_shape, row, col, rate):
    """
    Create the rate matrix.

    """
    m = create_sparse_rate_matrix(state_space_shape, row, col, rate)
    return m.A


def create_indicator_array(
        node,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations):
    """
    Create the initial array indicating observations.

    """
    nsites, nobservables = iid_observations.shape
    state_space_ndim = len(state_space_shape)
    state_space_axes = range(state_space_ndim)

    # Initialize the active array, initially in a high dimensional shape.
    # This array is large; for data with many iid sites,
    # such active arrays dominate the memory usage of the program.
    obs_shape = (nsites, ) + tuple(state_space_shape)
    obs = np.ones(obs_shape, dtype=float)

    # For each observable associated with the node under consideration,
    # apply the observation mask across all iid sites.
    local_observables = np.flatnonzero(observable_nodes == node)
    for idx in local_observables:
        states = iid_observations[:, idx]
        axis = observable_axes[idx]
        k = state_space_shape[axis]
        projection_shape = [k if i == axis else 1 for i in state_space_axes]
        mask_shape = (nsites, ) + tuple(projection_shape)
        ident = np.identity(k)
        obs *= np.take(ident, states, axis=0).reshape(mask_shape)

    # Reshape the observation array to 2d.
    # First collapse the dimensionality of the state space to 1d,
    # then transpose the array so that it has shape (nstates, nsites),
    # to prepare for P.dot(obs) where P has shape (nstates, nstates).
    obs = obs.reshape((nsites, np.prod(state_space_shape))).T

    # Return the observation indicator array.
    return obs


class EigenRateExpm(object):
    def __init__(self, Q_dense):
        #print('computing eigendecomposition...')
        self.w, self.U = eig(Q_dense)
        #print('inverting...')
        self.V = inv(self.U)
        #print('done preprocessing the rate matrix.')

    def get_P(self, rate_scaling_factor):
        w_exp = np.exp(self.w * rate_scaling_factor)
        P_raw = (self.U * w_exp).dot(self.V)
        return np.clip(P_raw.real, 0, np.inf)


def get_conditional_likelihoods(
        T, root, edges, edge_rate_pairs, edge_process_pairs,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations,
        processes_row,
        processes_col,
        processes_rate,
        ):
    """
    Recursively compute conditional likelihoods at the root.

    Attempt to order things intelligently to avoid using
    more memory than is necessary.

    The data provided by the caller gives us a sparse matrix
    of shape (nsites, nnodes, nstates).

    """
    child_to_edge = dict((tail, (head, tail)) for head, tail in edges)
    edge_to_rate = dict(edge_rate_pairs)
    edge_to_process = dict(edge_process_pairs)

    # For each process, precompute the eigendecomposition.
    eigen_rate_expm_objects = []
    nprocesses = len(processes_row)
    for i in range(nprocesses):
        row = processes_row[i]
        col = processes_col[i]
        rate = processes_rate[i]
        Q = create_dense_rate_matrix(state_space_shape, row, col, rate)
        obj = EigenRateExpm(Q)
        eigen_rate_expm_objects.append(obj)

    # For the few nodes that are active at a given point in the traversal,
    # we track a 2d array of shape (nsites, nstates).
    node_to_array = {}
    for node in get_node_evaluation_order(T, root):

        # When a node is activated, its associated array
        # is initialized to its observational likelihood array.
        arr = create_indicator_array(
                node,
                state_space_shape,
                observable_nodes,
                observable_axes,
                iid_observations)

        # When an internal node is activated,
        # this newly activated observational array is elementwise multiplied
        # by each of the active arrays of the child nodes.
        # The new elementwise product becomes the array
        # associated with the activated internal node.
        # The child nodes then become inactive and their
        # associated arrays are deleted.
        for child in T.successors(node):
            arr *= node_to_array[child]
            del node_to_array[child]

        # When any node that is not the root is activated,
        # the matrix product P.dot(A) replaces A,
        # where A is the active array and P is the matrix exponential
        # associated with the parent edge.
        if node != root:
            edge = child_to_edge[node]
            edge_rate = edge_to_rate[edge]
            edge_process = edge_to_process[edge]
            row = processes_row[edge_process]
            col = processes_col[edge_process]
            rate = processes_rate[edge_process]
            rate = rate * edge_rate

            # First way
            #Q = create_dense_rate_matrix(state_space_shape, row, col, rate)
            #P = expm(Q)
            #arr = P.dot(arr)

            # Second way
            #Q = create_sparse_rate_matrix(state_space_shape, row, col, rate)
            #arr = expm_multiply(Q, arr)

            # Third way
            P = eigen_rate_expm_objects[edge_process].get_P(edge_rate)
            arr = P.dot(arr)

        # Associate the array with the current node.
        node_to_array[node] = arr

    # The only active array should be the root at this point.
    # Check that invariant and return the array.
    assert_equal(len(node_to_array), 1)
    assert_equal(set(node_to_array), {root})
    return node_to_array[root]



def get_example_tree():
    T = nx.DiGraph()
    T.add_edges_from((
        (0, 1),
        (1, 2),
        (1, 3),
        (0, 4),
        (4, 5),
        (5, 6),
        (5, 7),
        (4, 8)))
    root = 0
    return T, root


def test_subtree_thickness():
    T, root = get_example_tree()
    d_actual = get_node_to_subtree_thickness(T, root)
    d_desired = {
            0 : 3,
            1 : 2,
            2 : 1,
            3 : 1,
            4 : 2,
            5 : 2,
            6 : 1,
            7 : 1,
            8 : 1}
    assert_equal(d_actual, d_desired)


def test_node_evaluation_order():
    T, root = get_example_tree()
    v_actual = list(get_node_evaluation_order(T, root))
    v_desired = (7, 6, 5, 8, 4, 3, 2, 1, 0)
    assert_equal(v_actual, v_desired)


def process_json_in(j_in):

    # Unpack some sizes and shapes.
    nnodes = j_in['node_count']
    nprocesses = j_in['process_count']
    state_space_shape = np.array(j_in['state_space_shape'])

    # Unpack stuff related to observables.
    info = get_observables_info(j_in)
    observable_nodes, observable_axes, iid_observations = info

    # Unpack stuff related to the prior distribution.
    info = get_prior_info(j_in)
    prior_feasible_states, prior_distribution = info

    # Unpack stuff related to the edge-specific processes.
    info = get_processes_info(j_in)
    processes_row, processes_col, processes_rate = info

    # Unpack stuff related to the tree and its edges.
    info = get_tree_info(j_in)
    T, root, edges, edge_rate_pairs, edge_process_pairs = info

    # Interpret the prior distribution.
    nstates = np.prod(state_space_shape)
    feas = np.ravel_multi_index(prior_feasible_states.T, state_space_shape)
    distn = np.zeros(nstates, dtype=float)
    np.put(distn, feas, prior_distribution)

    # Compute the conditional likelihoods.
    arr = get_conditional_likelihoods(
            T, root, edges, edge_rate_pairs, edge_process_pairs,
            state_space_shape,
            observable_nodes,
            observable_axes,
            iid_observations,
            processes_row,
            processes_col,
            processes_rate)

    # Apply the prior distribution and take logs of the likelihoods.
    likelihoods = distn.dot(arr)
    log_likelihoods = np.log(likelihoods)

    # Adjust for infeasibility.
    feasibilities = np.isfinite(log_likelihoods)
    log_likelihoods = np.where(feasibilities, log_likelihoods, 0)

    # Create the output in a format that json will like.
    j_out = dict(
            status = 'success',
            feasibilities = feasibilities.astype(int).tolist(),
            log_likelihoods = log_likelihoods.tolist())

    return j_out


def main(args):
    test_subtree_thickness()
    test_node_evaluation_order()

    try:
        s_in = sys.stdin.read()
        j_in = json.loads(s_in)
    except Exception as e:
        if args.debug:
            raise
        return dict(
                status = 'error',
                message = 'json parsing error: ' + traceback.format_exc())
    try:
        return process_json_in(j_in)
    except Exception as e:
        if args.debug:
            raise
        return dict(
                status = 'error',
                message = 'processing error: ' + traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    j_out = main(parser.parse_args())
    print(json.dumps(j_out))
