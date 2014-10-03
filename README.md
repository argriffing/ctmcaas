Example usage:

```
.../log-likelihoods$ cat jc.in.json
```
```json
{
	"tree" : {
		"order" : 2,
		"row" : [0],
		"col" : [1],
		"data" : [1]},
	"rates" : {
		"order" : 4,
		"row" : [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
		"col" : [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
		"data" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
	"prior" : [0.25, 0.25, 0.25, 0.25],
	"observable_nodes" : [0, 1],
	"iid_observations" : [
		[0, 0],
		[2, 2],
		[0, 1]]
}
```

```
.../log-likelihoods$ cat jc.in.json | python wrapped-npmctree.py | python -m json.tool
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
