# Predicting to Improve: Integrity Measures for Assessing Visual Localization Performance

This repo contains source code for our paper: "Predicting to Improve: Integrity Measures for Assessing Visual Localization Performance", available from the [publisher](https://ieeexplore.ieee.org/document/9830823) and on QUT [ePrints](https://eprints.qut.edu.au/234489/).

## Attribution

When using code within this repository, please reference the following [paper](https://ieeexplore.ieee.org/document/9830823):
```
@ARTICLE{9830823,
  author={Carson, Helen and Ford, Jason J. and Milford, Michael},
  journal={IEEE Robotics and Automation Letters}, 
  title={Predicting to Improve: Integrity Measures for Assessing Visual Localization Performance}, 
  year={2022},
  volume={7},
  number={4},
  pages={9627-9634},
  doi={10.1109/LRA.2022.3191205}}
```

## Installation

We recommend using conda or mamba to install all dependencies. Mamba can be installed from [`mambaforge`](https://github.com/conda-forge/miniforge).

```bash
conda create --name vpred_env python=3.9 numpy matplotlib jupyterlab scikit-learn -c conda-forge
conda activate vpred_env
```

Download the example Nordland feature set using the link [here](https://cloudstor.aarnet.edu.au/plus/s/UgpN69h5VP2thG8).

Note these features are derived from the partitioned Nordland testset published at http://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset by David Olid et al in **Single-View Place Recognition under Seasonal Changes** *In PPNIV Workshop at IROS 2018*.

Run the jupyterlab example notebook using:
```
jupyter lab example.ipynb
```

## Licence

The code is licensed under the [MIT License](./LICENSE).
