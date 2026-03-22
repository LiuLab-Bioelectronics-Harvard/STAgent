# STAligner

![STAligner_Overview](https://github.com/zhoux85/STAligner/assets/31464727/1358f6b0-75ed-4bdd-9d0b-257788dff73a)


## Overview

STAligner is designed for alignment and integration of spatially resolved transcriptomics data.

**a**. STAligner first normalizes the expression proﬁles for all spots and constructs a spatial neighbor network using the spatial coordinates. STAligner further employs a graph attention auto-encoder neural network to extract spatially aware embedding, and constructs the spot triplets based on current embeddings to guide the alignment process by attracting similar spots and discriminating dissimilar spots across slices. STAligner introduces the triplet loss to update the spot embedding to reduce the distance from the anchor to positive spot, and increase the distance from the anchor to negative spot. The triplet construction and auto-encoder training are optimized iteratively until batch-corrected embeddings are generated. **b**. STAligner can be applied to integrate ST datasets to achieve alignment and simultaneous identification of spatial domains from different biological samples in (**a**), technological platforms (I), developmental (embryonic) stages (II), disease conditions (III) and consecutive slices of a tissue for 3D slice alignment (IV).



## Installation
The STAligner package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) (*PyTorch Geometric*) framework, and can be run on GPU (recommend) or CPU.



First clone the repository. 

```
git clone https://github.com/zhoux85/STAligner.git
cd STAligner-main
```

It's recommended to create a separate conda environment for running STAligner:

```
#create an environment called env_STAligner
conda create -n env_STAligner python=3.8

#activate your environment
conda activate env_STAligner
```

Install all the required packages. 

For Linux
```
pip install -r requirement.txt
```
For MacOS
```
pip install -r requirement_for_macOS.txt
```

The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

The torch-geometric library is also required, please see the installation steps in https://github.com/pyg-team/pytorch_geometric#installation

Install STAligner.

```
python setup.py build
python setup.py install
```



## Tutorials

Three step-by-step tutorials are included in the `Tutorial` folder and https://staligner.readthedocs.io/en/latest/ to show how to use STAligner. 

- Tutorial 1: Integrating 4 adjacent DLPFC slices (10x Visium)
- Tutorial 2: Integrating all 12 DLPFC slices from 3 adult samples (10x Visium)
- Tutorial 3: Integrating slices across sequencing platforms (Slide-seqV2 and Stereo-seq)
- Tutorial 4: Integrating 4 mouse embryo slices sampled at the time stages of E9.5, E10.5, E11.5, and E12.5 (Stereo-seq)
- Tutorial 5: Spatial domain guided 3D slices alignment (Slide-seq)



## Support

If you have any questions, please feel free to contact us [xzhou@amss.ac.cn](mailto:xzhou@amss.ac.cn). 



## Citation
Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nat Comput Sci 3, 894–906 (2023). https://doi.org/10.1038/s43588-023-00528-w

