#  Python codes for ***vertex upwinding scheme*** implemented in FEniCS and PETSc
Numerical examples for
> Mohammad. S. Joshaghani and Beatrice Riviere,
> ``A computational framework for a vertex unwinding scheme in two-phase flow"

In this repoository, we have provided python computer codes for the numerical solution of immiscible two-phase flows in porous media, which is obtained by a first order finite element method equipped with mass-lumping and flux upwinding.
This code entails several examples of quarter-five spot problems in two and three dimensions. We also illustrated how to handle heterogeneities in the permeability field. More details are discussed in the paper.

## Project tree
2D_Heterogeneous
└── vertex_heterogen.py
2D_Homogeneous
├── DG_structured.py
├── Unstructured1.xml
├── Vertex_structured.py
└── Vertex_unstructured.py
3D_Homogeneous
├── main_vertex.py
└── mesh.xml
Convergence
└── Vertex_convergence.py
SPE
├── 2D
│   ├── K_Input
│   │   ├── Kx.mat
│   │   ├── Kx_1.mat
│   │   ├── Kx_45.mat
│   │   └── Kx_80.mat
│   └── spe10.py
└── 3D
    ├── K_Input
    │   ├── .DS_Store
    │   └── layers.mat
    └── spe10.py



## Runscript
First install Docker CE for your platform (Windows, Mac or Linux) and then run the following command in your bash:
```bash
docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable
```
Docker container is started and you can run all the codes.
