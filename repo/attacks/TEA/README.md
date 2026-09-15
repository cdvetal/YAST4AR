This directory contains the implementation and tutorial for the Targeted Edge-informed Attack (TEA).

## Installation

1. **Clone the repo** (if you haven’t already):

   ```bash
   git clone https://github.com/mdppml/TEA.git
   cd TEA/ATTACK
   ```

2. **Create & activate the conda environment**:

   ```bash
   conda env create -f conda_environment.yml
   conda activate tea
   ```

## Contents

* **TEA.py**
  Implements the core methodology and contains four important functions:

  1. `initialize_attack(...)` — displays source & target images and the $\ell^2$ distance between them.
  2. `edge_mask_initialization(...)` — visualizes the edge mask and soft-edge mask of the target image.
  3. `global_edge_informed_search(...)` — applies global perturbations while preserving edge regions.
  4. `patch_based_edge_informed_search(...)` — applies local patch-wise perturbations while preserving edge regions.

* **utils.py**
  Utility functions for model/device setup, image loading & transformation, and image-pair management.

* **conda\_environment.yml**
  Environment file containing package information.
  
* **tutorial.ipynb**
  Jupyter notebook demonstrating how to:

  1. Run TEA.
  2. Visualize edge masks.
  3. Visualize perturbations corresponding to both the global edge-informed search and patch-based edge-informed search.


