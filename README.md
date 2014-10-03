Example usage:

```
$ cat jc.in.json | python log-likelihood.py | python -m json.tool
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
