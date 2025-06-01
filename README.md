## Supplement To "Deep learning-based morphological classification of ceramics: A case study of 3D Point Cloud Analysis for Sue ware, Japan"

[![DOI](https://zenodo.org/badge/989476385.svg)](https://doi.org/10.5281/zenodo.15523451)

### Contents

```sh
.
├── CeramicNet+PointTransformer.py           # stand-alone Python script for analysis
├── mainscript.ipynb                         # Jupyter notebook version of the workflow
├── requirements.txt                         # Python dependencies for supplement
├── docker                                   # Dockerfiles for reproducible environments
│   ├── cpu.Dockerfile                       # CPU-only execution image
│   ├── gpu.Dockerfile                       # CUDA-enabled execution image
│   └── mainscript.Dockerfile                # image that executes the notebook headlessly (CPU-only)
├── ceramicnet_data                          # primary Sue ware dataset (1024 pts / sample)
│   ├── ceramicnet_shape_names.txt           # list of class names
│   ├── B                                    # Bowl samples
│   ├── DB                                   # Dish Body samples
│   ├── DBR                                  # Dish Body with Ring Base samples
│   ├── DC                                   # Dish Cap samples
│   └── P                                    # Plate samples
├── other_data                               # additional datasets for further experiments
│   └── DCFLIP                               # upside-down DC samples used in dcflip_analysis
└── results_used_in_article                  # materials used in the mainscript
    ├── main_analysis                        # main experiment outputs
    │   ├── mainscript_executed.ipynb        # Table 1 and 2 are available in this notebook
    │   ├── accuracy_epoch*.png              # accuracy curves for each fold; not used in the mainscript
    │   ├── pca_epoch*.png                   # PCA plots used in Figure 6
    │   ├── dendrogram_epoch*.png            # hierarchical clustering dendrograms used in Figure 7
    │   └── gradcam_*.png                    # Grad-CAM visualizations used in Figure 8
    └── dcfilp_analysis                      # DCFLIP experiment outputs
        ├── dcflip_analysis_executed.ipynb   # mentioned in Section 6.2
        ├── accuracy_epoch*.png              # not used in the mainscript
        ├── pca_epoch*.png                   # not used in the mainscript
        ├── dendrogram_epoch*.png            # not used in the mainscript
        └── gradcam_*.png                    # not used in the mainscript
```

### How to start analysis

The analysis can be run using either CPU or GPU support. The mode is automatically set based on the Dockerfile used.

#### CPU Version

To run the analysis using Docker with CPU support, follow these steps:

1. Create an output directory on your host machine to store the output files

   ```bash
   mkdir -p output
   ```

2. Build the Docker image

   ```bash
   docker build -t ceramicnet-cpu -f docker/cpu.Dockerfile .
   ```

3. Run the container with a volume mount

   ```bash
   # bash:
   docker run --rm -v $(pwd)/output:/app/output ceramicnet-cpu

   # PowerShell:
   docker run --rm -v ${PWD}/output:/app/output ceramicnet-cpu
   ```

#### GPU Version

To run the analysis using Docker with GPU support, follow these steps:

1. Ensure you have NVIDIA Container Toolkit installed and your GPU drivers are up to date.

2. Create an output directory on your host machine to store the output files

   ```bash
   mkdir -p output
   ```

3. Build the Docker image

   ```bash
   docker build -t ceramicnet-gpu -f docker/gpu.Dockerfile .
   ```

4. Run the container with GPU support and volume mount

   ```bash
   # bash:
   docker run --rm --gpus all -v $(pwd)/output:/app/output ceramicnet-gpu

   # PowerShell:
   docker run --rm --gpus all -v ${PWD}/output:/app/output ceramicnet-gpu
   ```

The container will automatically execute the analysis script and save all output files (plots, metrics, etc.) to the `output` directory on your host machine. The `--rm` flag ensures the container is removed after execution.

Note: We used the following GPU to confirm the script.

```sh
$ nvidia-smi
Sun Jan  7 23:40:57 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.33                 Driver Version: 546.33       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3070 ...  WDDM  | 00000000:01:00.0  On |                  N/A |
| N/A   47C    P8              19W / 130W |    720MiB /  8192MiB |     28%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:30:42_Pacific_Standard_Time_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```

### Running Jupyter Notebook with Docker

To run the Jupyter Notebook analysis using Docker, follow these steps:

1. Create an output directory on your host machine to store the output files

   ```bash
   mkdir -p output
   ```

2. Build the Docker image for the notebook

   ```bash
   docker build -t ceramicnet-notebook -f docker/mainscript.Dockerfile .
   ```

3. Run the container with a volume mount

   ```bash
   # bash:
   docker run --rm -v $(pwd)/output:/app/output ceramicnet-notebook

   # PowerShell:
   docker run --rm -v ${PWD}/output:/app/output ceramicnet-notebook
   ```

The container will automatically execute the notebook and save:

- The executed notebook as `mainscript_executed.ipynb` in the container
- All output files (plots, metrics, etc.) to the `output` directory on your host machine
- The `--rm` flag ensures the container is removed after execution

Note: The notebook execution may take some time depending on your system's resources. All results will be saved in the `output` directory.

### Primary References for Implementation

Zhao, H., Jiang, L., Jia, J., Torr, P. H., & Koltun, V. (2021). Point transformer. Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV), 16259–16268.

Matrone, F., Felicetti, A., Paolanti, M., & Pierdicca, R. (2023). Explaining ai: Understanding deep
learning models for heritage point clouds. ISPRS Annals of the Photogrammetry, Remote Sensing
and Spatial Information Sciences, X-M-1-2023, 207–214. https://doi.org/10.5194/isprs-annals-
X-M-1-2023-207-2023

### License

| Asset                           | License   | File            |
| ------------------------------- | --------- | --------------- |
| Source code                     | MIT       | LICENSE         |
| Data, figures, notebook outputs | CC BY 4.0 | LICENSE-DATA.md |

If you use this code or dataset, please cite the accompanying publication (see `CITATION.cff`).
