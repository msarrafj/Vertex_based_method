<center><h1>Codes for vertex upwinding scheme implemented in FEniCS and PETSc</h1></center>

Numerical examples for
> Mohammad. S. Joshaghani and Beatrice Riviere,
> ``A computational framework for a vertex unwinding scheme in two-phase flow" Available in [arXiv](https://arxiv.org/abs/2103.03285).
> <details><summary>[Abstract]</summary>
><p> This paper presents the numerical solution of immiscible two-phase flows in porous media, obtained by a first-order finite element method equipped with mass-lumping and flux up-winding. The unknowns are the physical phase pressure and phase saturation. Our numerical experiments confirm that the method converges optimally for manufactured solutions. For both structured and unstructured meshes, we observe the high-accuracy wetting saturation profile that ensures minimal numerical diffusion at the front. Performing several examples of quarter-five spot problems in two and three dimensions, we show that the method can easily handle heterogeneities in the permeability field. Two distinct features that make the method appealing to reservoir simulators are: (i) maximum principle is satisfied, and (ii) mass balance is locally conserved.
></p>
></details>

In this repoository, we have provided python computer codes for the numerical solution of immiscible two-phase flows in porous media, which is obtained by a first order finite element method equipped with mass-lumping and flux upwinding.
This code entails several examples of quarter-five spot problems in two and three dimensions. We also illustrated how to handle heterogeneities in the permeability field. More details are discussed in the paper.

## Representative results
<img src="Video/Video1.gif" width="800" />
<!-- ![](./Video/Video1.gif) -->
<p align="center">
This video shows the evolution of saturation solutions in a two-dimension porous medium with a low permeability block located in the center of domain. Two cases are compared: on the left, the block is one-order less permeable (i.e., K/K<sub><i>In</i></sub>=10); on the right, the block is four-order less permeable (i.e., K/K<sub><i>In</sub>=10000). In first case, fluid initially evades the block but as time progresses saturation inside the block starts to increase. However, for the second case, higher permeability difference has made the inclusion impenetrable throughout the simulation. We also observe that the proposed vertex scheme delivers satisfactory results with respect to maximum principle (saturation remains between 0.15 and 0.85). For more details on BVP and parameters refer to section 5.2.4 of the paper.
</p>

<p align="center">
<img src="Video/Video2.gif" width="600" />
</p>

<p align="center"> 

This video shows the evolution of the saturation solution in highly heterogeneous porous medium (layer 45 of SPE10 problem). The corresponding permeability field can be found [here](./Video/Figure1.png). It is evident that permeability field dictates the pattern fluid flows through porous medium. Furthermore, the video reiterates that the proposed vertex scheme always produce physical values (between 0.15 and 0.85 in this problem), without any undershoots and overshoots, even for domains with permeabilities that vary over many orders of magnitude. For more details on this simulation, refer to section 5.2.5 of the paper. 
</p>


![](./Video/Video3.gif)
<p align="center">

This video shows contours of the saturation solutions obtained under the proposed vertex scheme on a highly heterogeneous domain (i.e., section of SPE10 benchmark paroblem). We can observe that the proposed vertex scheme generates robust and accurate outputs. The wetting phase fluid flows through the most permeable pore-networks from infection to production well (see [here](./Video/Figure2.png) for permeability field). In addition, the scheme respects the maximum principle, since no undershoots and overshoots has been observed during the simulation.  For more details on this problem, refer to section 5.2.7 of the paper.
</p>

## Project tree
```
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
 ```



## Runscript
First install Docker CE for your platform (Windows, Mac or Linux) and then run the following command in your bash:
```bash
docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable
```
Docker container is started and you can run all the codes.
