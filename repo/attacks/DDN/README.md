
# DDN attack

## How to run:
The attack can be executed by running the test.py file inside the DDN folder.
There are 3 ways to run the attack:


* Running using the test.py values:

```console
    python3 attacks\DDN\DDN.py
```


* Running using a config file:

```console
    python3 attacks\DDN\DDN.py --json-config config.json
```

* Specifying all parameters

```console
    python3 attacks\DDN\test.py --steps 1000 --gamma 0.05 --init_norm 1.0 --quantize --levels 256 
```

**Note:** Its possible to pass a config file or use the default values in the test file and override the default values by specifying the parameters in the command.



## Parameters:

| Parameter          | Range\|\[Options\]        | Type   | Description                                                                                    |
|--------------------|:-------------------------:|:------:|:-----------------------------------------------------------------------------------------------|
| steps              | 0 < n                     | int    | Number of steps for the optimization                                                           |
| targeted           |                           | bool   | Whether to perform a targeted attack or not                                                    |
| gammma             |                           | float  | Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma)                  |
| init-norm          | 0 < n                     | float  | Initial value for the norm                                                                     |
| quantize           |                           | bool   | If True, the returned adversarials will have quantized values to the specified number of levels|
| levels             | 0 < n                     |        | Number of levels to use for quantization (e.g. 256 for 8 bit images)                           |
| path-output-folder |                           | str    | path to store results                                                                          |
| log                |                           | bool   | log process during attack                                                                      |
| total-images       | 0 < n                     | int    |                                                                                                |
| batch-size         | 0 < n                     | int    | total-images must be divisible by batch-size                                                   |
| json-config        |                           | str    | config file to be passed as default parameters                                                 |
 


