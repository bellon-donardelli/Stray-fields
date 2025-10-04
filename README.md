# Stray-Fields — TetraModels

This repository provides scripts and data for computing the **magnetic stray field** of individual particles from micromagnetic solutions generated with **MERRILL**.  
It also includes utilities to simulate **Quantum Diamond Microscope (QDM)** noise and to generate **QDM-like maps** from tetrahedral finite-element solutions.

---

## Features

- **Stray Field Solver**: Uses a **tetrahedron potential-field formulation** on finite-element micromagnetic solutions.  
- **QDM Map Generation**: Produces QDM-like magnetic maps at user-specified stand-off heights.  
- **QDM Noise Simulation**: Adds realistic QDM-like noise to generated maps.  
- **Example Data**: Includes `.tec` and the generated `.nc` files for quick testing and validation.  

---

## Quick Start

1. **Clone this repository**
   ```bash
   gh repo clone bellon-donardelli/Stray-fields
   cd Stray-fields/TetraModels

## Contact 

For questions or collaboration:
ualisson.bellon@ed.ac.uk

## Data Repository and Citation

If you use this repository, please cite:
 
[https://doi.org/10.5281/zenodo.17256342](https://doi.org/10.5281/zenodo.17256342)



## Authors

- U.D. Bellon, G.F. Souza-Junior, W. Williams, L. Uieda, A.R. Muxworthy, L. Nagy, G.A. Paterson, C.S. Moraes, R. Fu, R.I.F. Trindade. 
