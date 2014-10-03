"""
Implement log likelihoods for complicated models.

This interface takes some care about memory usage,
while allowing more subtlety in the representation of observed data,
and while allowing more flexibility in the representation of
inhomogeneity of the process across branches.
{
	"nodes" : 2,
	"state_space" : [4, 1],
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
