# BanditsPrior attack

## How to run:
The attack can be executed by running the test.py file inside the BanditsPrior folder.

It's also possible to run the BanditsPrior.py file directly by passing all the necessary parameters or a config.json.
Command with all parameters:
```console
    python3 attacks/BanditsPrior/BanditsPrior.py --max-queries 200 --fd-eta 0.01 --image-lr 0.5 --online-lr 0.1 --mode l2 --exploration 0.01 --tile-size 50 --epsilon 5.0 --batch-size 10 --gradient-iters 1 --device cuda --path-output-folder outputs\test
```
Command passing a config.json file:
```console
    python3 attacks/BanditsPrior/BanditsPrior.py --json-config config.json
```

The config.json file are considered as default values. If other value is set in the command, the config.json value will be ignored.


## Parameters:

| Parameter          | Range\|\[Options\]      | Type   | Facultative | Description                                                     |
|--------------------|:-----------------------:|:------:|:-----------:|----------------------------------------------------------------:|
| max-queries        | 0 < n                   | int    | F           |                                                                 |
| fd-eta             | 0 < n <=1               | float  | F           | Η, used to estimate the derivative via finite differences       |
| image-lr           | 0 < n                   | float  | F           | Learning rate for the image (iterative attack)                  |
| online-lr          | 0 < n                   | float  | F           | Learning rate for the prior                                     |
| mode               | \[linf\|l2\]            | str    | F           | Which lp constraint to run bandits                              |
| exploration        | 0 < n                   | float  | F           | Δ, parameterizes the exploration to be done around the prior    |
| tile-size          | 0 < n                   | int    | F           | the side length of each tile (for the tiling prior)             |
| json-config        |                         | str    | T           | a config file to be passed in instead of arguments              |
| epsilon            | 0 < n                   | float  | F           |                                                                 |
| batch-size         | 0 < n                   | int    | F           |                                                                 |
| log-progress       |                         |        | T           |                                                                 |
| nes                |                         |        | T           |                                                                 |
| tiling             |                         |        | T           |                                                                 |
| gradient-iters     | 0 < n                   | int    | T           |                                                                 |
| device             | \[cpu\|cuda\]           | str    | T           | device to run the attack                                        |
| path-dataset       |                         | str    | T           | path to load dataset (TO BE IMPLEMENTED)                        |
| path-model         |                         | str    | T           | path to load model (TO BE IMPLEMENTED)                          |
| path-output-folder |                         | str    | T           | path to store results                                           |
