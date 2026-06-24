# CVR for Very Large Regression Problems

*Read this in other languages: [Tiếng Việt](README-vi.md)*

This repository contains the official implementation and evaluation notebooks for Core Vector Regression (CVR) and Core Vector Machine (CVM), specifically optimized for very large-scale problems.

## Citation

> Ivor W. Tsang, James T. Kwok, and Kimo T. Lai. 2005. Core Vector Regression for very large regression problems. In Proceedings of the 22nd international conference on Machine learning (ICML '05). Association for Computing Machinery, New York, NY, USA, 912–919. [https://doi.org/10.1145/1102351.1102466](https://doi.org/10.1145/1102351.1102466)

## Directory Structure

- **`implement/`**: Contains the highly optimized C++ source code for the models.
  - `cvm_train.cpp`: Implementation of the Core Vector Machine for classification tasks.
  - `cvr_train.cpp`: Implementation of the Core Vector Regression for regression tasks.
  *Note: Both implementations utilize SMO (Sequential Minimal Optimization) and are heavily optimized with AVX2/BMI instruction sets for maximum performance. They read standard LIBSVM format files.*

- **`evaluation/`**: Contains Jupyter notebooks used to benchmark the performance of the implementations.
  - `large-benchmark-data-sets/`:
    - `cvr-full.ipynb`: Comprehensive notebook for evaluating the models on all standard large-scale OpenML datasets.
    - `parts/`: Contains broken-down versions of `cvr-full.ipynb` running specific experiments to avoid timeout or memory issues on cloud environments.

      **Evaluated Models Mapping**:
      - **CVR**: L2-SVR (Core Vector Regression)
      - **LIBSVM**: L1-SVR (LIBSVM)
      - **SVMlight**: L1-SVR (SVM^light)
      - **CVM**: L2-SVM (Core Vector Machine)
      - **nu-SVR**: $\nu$-SVR (LIBSVM)

      **Regression Benchmark Notebooks**

      | Notebook File | Datasets | Models Compared | Data Splits (%) |
      | --- | --- | --- | --- |
      | `01_standard_benchmarks.ipynb` | `census_housing`, `computer_activity`, `elevators`, `pole_telecomm` | CVR, LIBSVM, SVMlight | 10, 30, 50, 90 |
      | `02_novel_datasets.ipynb` | `superconductivity`, `bike_sharing` | CVR, LIBSVM, SVMlight | 10, 30, 50, 90 |
      | `03_friedman_cvr_vs_libsvm.ipynb` | `friedman` | CVR, LIBSVM | 5, 10, 20, 35, 50, 70, 90 |
      | `04_friedman_all_models.ipynb` | `friedman` | CVR, LIBSVM, SVMlight | 5, 10, 20, 35, 50 |

      **Classification Benchmark Notebooks**

      | Notebook File | Datasets | Models Compared | Data Splits (%) |
      | --- | --- | --- | --- |
      | `05_covertype_nusvr.ipynb` | `covertype` | nu-SVR | 1, 2, 5, 10, 20 |
      | `06_covertype_cvm_cvr.ipynb` | `covertype` | CVM, CVR | 1, 2, 5, 10, 20, 30, 50, 70, 90 |

  - `surface-modeling/`: Notebooks for evaluating the models on 3D implicit surface reconstruction tasks using multi-scale Core Vector Regression on point cloud data.
    - `01_stanford_bunny_multiscale.ipynb`: Surface modeling for the Stanford Bunny dataset.
    - `02_stanford_dragon_multiscale.ipynb`: Surface modeling for the Stanford Dragon dataset.

- **`ablation/`**: Contains Jupyter notebooks for ablation studies. These notebooks systematically disable or alter specific hyperparameters and components (e.g., varying epsilon, mu, gamma, or modifying the SMO solver steps) to measure their impact on model size, convergence time, and accuracy.
  - `01_ablation_hyperparams_components_benchmark.ipynb`: Ablation study of hyperparameters and components on standard benchmark datasets.
  - `02_ablation_hyperparams_components_surface.ipynb`: Ablation study of hyperparameters and components specifically on 3D surface modeling tasks.

## Datasets Information

### Benchmark Datasets

| Dataset | Task Type | Features | Instances (N) |
| --- | --- | --- | --- |
| `census_housing` | Regression | 121 | 22,732 |
| `computer_activity` | Regression | 21 | 8,192 |
| `elevators` | Regression | 18 | 16,599 |
| `friedman` | Regression | 10 | 100,000 |
| `pole_telecomm` | Regression | 48 | 15,000 |
| `superconduct` | Regression | 81 | 21,263 |
| `bike_sharing` | Regression | 12 | 17,379 |
| `covertype` | Classification | 54 | 581,012 |

### Surface Modeling Datasets & Parameters

| Dataset | Object Type | Kernel | Sigmas (Scales) | C | mu | epsilon |
| --- | --- | --- | --- | --- | --- | --- |
| `Stanford Bunny` | 3D Point Cloud (10,000 pts) | Laplacian | [0.2, 0.1, 0.05] | 1.0 | 1e-4 | 1e-4 |
| `Stanford Dragon` | 3D Point Cloud (30,000 pts) | Laplacian | [0.1, 0.05, 0.02] | 10.0 | 1e-4 | 1e-5 |

*Note: The surface modeling uses a Laplacian Kernel (`exp(-gamma * ||x-y||)`) and trains iteratively across multiple scales (`gamma = 1/sigma`) to progressively capture crude shapes, medium details, and fine details.*

## Setup & Compilation (Local Environment)

The C++ source files in the `implement/` directory can be compiled using `g++` with high optimization flags. For example:

```bash
g++ -O3 -std=c++11 implement/cvm_train.cpp -o cvm_train
g++ -O3 -std=c++11 implement/cvr_train.cpp -o cvr_train
```

The compiled binaries expect arguments such as the path to the training data (in LIBSVM format), hyperparameter values (C, gamma, epsilon, etc.), and the output path. Please refer to the source code for the exact argument order.

## Running Notebooks on Kaggle/Colab

The notebooks in `evaluation/` and `ablation/` were designed to run on cloud platforms like Kaggle and Google Colab, leveraging their environment and hardware. **They will not run out-of-the-box locally** because they contain cloud-specific absolute paths (`/kaggle/working/...`) and system-level commands to compile baseline libraries (`libsvm`, `SVM^light`, `libCVM`).

To run these notebooks:

1. **Kaggle**:
   - Create a new notebook on Kaggle.
   - Upload the required notebook file from this repository using the `File -> Import Notebook` option.
   - Run the cells. The notebook will automatically download the datasets, compile the necessary binaries, and execute the experiments.
2. **Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/).
   - Select `Upload` and choose the notebook file you want to run.
   - Run the cells. The notebook will execute the experiments inside the Colab environment.

> [!IMPORTANT]  
> **Path Configurations:** The default working directory on Kaggle is `/kaggle/working/`, while on Google Colab it is `/content/`. Depending on which platform a notebook was originally exported from, you might see hardcoded paths.
>
> - If you are on **Colab** and see `/kaggle/working/` in the code, replace it with `/content/`.
> - If you are on **Kaggle** and see `/content/` in the code, replace it with `/kaggle/working/`.
