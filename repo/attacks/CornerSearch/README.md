
# CornerSearch attack

## How to run:
The attack can be executed by running the test.py file inside the CornerSearch folder.

It's also possible to run the CornerSearch.py file directly by passing all the necessary parameters or a config.json.
Command with all parameters:
```console
    python3 attacks/CornerSearch/CornerSearch.py --type-attack L0+sigma --n-iter 1000 --n-max 100 --epsilon -1 --kappa -1 --sparsity 10 --size-incr 1 --device cuda --path-output-folder outputs\CornerSearchResults --log    
```
Command passing a config.json file:
```console
    python3 attacks/CornerSearch/CornerSearch.py --json-config config.json --log
```

The config.json file are considered as default values. If other value is set in the command, the config.json value will be ignored.


## Parameters:

| Parameter          | Range\|\[Options\]        | Type   | Facultative | Description                                                                                    |
|--------------------|:-------------------------:|:------:|:-----------:|-----------------------------------------------------------------------------------------------:|
| type_attack        | \[L0\|L0+Linf\|L0+sigma\] | str    | F           |                                                                                                |
| n_iter             | 0 < n                     | int    | F           | Number of iterations (N_iter in the paper)                                                     |
| n_max              | 0 < n                     | int    | F           | The modifications for k-pixels perturbations are sampled among the best n_max (N in the paper) |
| epsilon            |                           | int    | F           | For L0+Linf, the bound on the Linf-norm of the perturbation                                    |
| kappa              |                           | int    | F           | For L0+sigma (see kappa in the paper), larger kappa means easier and more visible attacks      |
| sparsity           |                           |        | F           | Maximum number of pixels that can be modified (k_max in the paper)                             |
| size-incr          |                           |        | F           | Size of progressive increment of sparsity levels to check                                      |
| batch-size         |                           |        | F           | Number of adversarial examples                                                                 |
| json-config        |                           | str    | T           | a config file to be passed in instead of arguments                                             |
| device             | \[cpu\|cuda\]             | str    | T           | device to run the attack                                                                       |
| path-output-folder |                           | str    | T           | path to store results                                                                          |
| log                |                           |        | T           |                                                                                                |

#batch_size; 

The paper meantioned it available in docs/sparse-imperceivable-long.pdf

