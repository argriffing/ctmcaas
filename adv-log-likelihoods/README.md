This is a more advanced interface for log likelihoods calculations.

The directory also includes models, data, and inference code for
analysis of an hky gene conversion model.

Example hky gene conversion log likelihood calculation:
```
$ python make-hky-geneconv-json.py --fasta=cleaned.fasta | python ll.py | python json-sum.py log_likelihoods
-1721.78341305
```

Example usage, for a toy model:
```
$ cat jc.in.json
```
```json
{
	"node_count" : 2,
	"process_count" : 1,
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
	"prior_feasible_states" : [[0, 0], [1, 0], [2, 0], [3, 0]],
	"prior_distribution" : [0.25, 0.25, 0.25, 0.25],
	"observable_nodes" : [0, 0, 1, 1],
	"observable_axes" : [0, 1, 0, 1],
	"iid_observations" : [
		[0, 0, 0, 0],
		[2, 0, 2, 0],
		[0, 0, 1, 0]]
}
```

```
$ cat jc.in.json | python ll.py | python -m json.tool
```
```json
{
    "feasibilities": [
        1,
        1,
        1
    ],
    "log_likelihoods": [
        -2.719098272533848,
        -2.719098272533848,
        -2.791074169065668
    ],
    "status": "success"
}
```
