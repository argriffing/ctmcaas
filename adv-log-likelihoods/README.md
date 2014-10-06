This is a more advanced interface for log likelihoods calculations.

The directory also includes models, data, and inference code for
analysis of an hky gene conversion model.

Example hky gene conversion log likelihood calculation:
```
$ python make-hky-geneconv-json.py --fasta=cleaned.fasta | python ll.py | python json-sum.py log_likelihoods
-1721.78341305
```


Example mle search using an internet log likelihood server:
```
$ python serve-ll.py
$ python hky-geneconv-mle.py --fasta=cleaned.fasta --ll_url=http://localhost:8080/
optimization result:
  status: 0
 success: True
    nfev: 480
     fun: 1721.7834131263585
       x: array([-1.26698319, -1.3697533 , -1.57771381, -1.37281197,  0.74847448,
        0.59884455, -2.65064603, -2.27328267, -2.97316305, -4.73377   ,
       -4.51730145, -3.51105446, -5.2956917 , -5.39206932])
 message: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
     jac: array([ 0.00090949,  0.0005457 , -0.00406999,  0.00231921, -0.00040927,
       -0.00179625, -0.0003638 , -0.00156888, -0.00056843,  0.00015916,
        0.00097771,  0.00115961,  0.00272848, -0.00552518])
     nit: 27

kappa: 2.11377294482
tau: 1.82001465591
nt_probs: [ 0.28289949  0.25526994  0.20734023  0.25449034]
edge rates:
('N0', 'N1') : 0.0706055846813
('N0', 'Tamarin') : 0.102973596402
('N1', 'Macaque') : 0.0511412917139
('N1', 'N2') : 0.00879325793232
('N2', 'N3') : 0.0109184479513
('N2', 'Orangutan') : 0.0298654060709
('N3', 'Gorilla') : 0.00501314556266
('N3', 'Chimpanzee') : 0.00455254289649
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
