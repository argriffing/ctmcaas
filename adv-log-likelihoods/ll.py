"""
Implement log likelihoods for complicated models.

This interface takes some care about memory usage,
while allowing more subtlety in the representation of observed data,
and while allowing more flexibility in the representation of
inhomogeneity of the process across branches.
{
	"nodes" : 2,
	"state_space_shape" : [4, 1],
	"tree" : {
		"row" : [0],
		"col" : [1],
		"rate" : [1],
		"process" : [0]},
	"processes" : [ {
		"row" : [
			[0, 0], [0, 0], [0, 0],
			[1, 0], [1, 0], [1, 0],
			[2, 0], [2, 0], [2, 0],
			[3, 0], [3, 0], [3, 0]],
		"col" : [
			[1, 0], [2, 0], [3, 0],
			[0, 0], [2, 0], [3, 0],
			[0, 0], [1, 0], [3, 0],
			[0, 0], [1, 0], [2, 0]],
		"rate" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] } ],
	"prior" : {
		"feasible_states" : [[0, 0], [1, 0], [2, 0], [3, 0]],
		"distribution" : [0.25, 0.25, 0.25, 0.25]},
	"observable_nodes" : [0, 1],
	"iid_observations" : [
		[[0, 0], [0, 0]],
		[[2, 0], [2, 0]],
		[[0, 0], [1, 0]]]
}

"""
from __future__ import print_function, division

import networkx as nx
import numpy as np
from numpy.testing import assert_equal


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


def create_dense_obs_array(node, state_space_shape, iid_observations):
    # TODO undo the compound state indexing as follows
    # TODO actually this would not work because of the partial observations
    #nsites, nnodes, state_space_ndim = iid_observations.shape
    #iid_observations = np.ravel_multi_index(
            #np.rollaxis(iid_observations, 2), state_space_shape)

    nsites, nnodes_observable = iid_observations.shape


def get_log_likelihoods(T, root):
    """
    Recursively compute partial likelihoods.

    Attempt to order things intelligently to avoid using
    more memory than is necessary.

    The data provided by the caller gives us a sparse matrix
    of shape (nsites, nnodes, nstates).

    """
    all_successors = T.successors()
    all_predecessors = T.predecessors()

    # For the few nodes that are "active" at a given point in the traversal,
    # we track a 2d array of shape (nsites, nstates).
    node_to_array = {}
    for n in get_node_evaluation_order(T, root):
        # When a node is activated,
        # the observational likelihood array is unpacked into the
        # array associated with the node.
        # If the node is unobservable then the array is filled with ones.

        # When an internal node is activated,
        # this newly activated observational array is elementwise multiplied
        # by each of the active arrays of the child nodes.
        # The new elementwise product becomes the array
        # associated with the activated internal node.
        # The child nodes then become inactive and their
        # associated arrays are deleted.

        # When any node that is not the root is activated,
        # the matrix product P.dot(A) replaces A,
        # where A is the active array and P is the matrix exponential
        # associated with the parent edge.
        """
	"iid_observations" : [
		[[0, 0], [0, 0]],
		[[2, 0], [2, 0]],
		[[0, 0], [1, 0]]]
        """

        pass


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


def test_high_water_mark():
    T, root = get_example_tree()
    d_actual = get_node_to_subtree_high_water_mark(T, root)
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


def main():
    test_high_water_mark()
    test_node_evaluation_order()
    pass


if __name__ == '__main__':
    main()
