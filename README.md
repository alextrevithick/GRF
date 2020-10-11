# GRF
This is the official repository for General Radiance Field (GRF) from:  
**GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering**  
[Alex Trevithick](https://alextrevithick.github.io/) [Bo Yang](https://yang7879.github.io/)  
\[Paper\] \[Code (coming soon)\] \[[Video (Download for full 1008 by 756 resolution)](https://drive.google.com/file/d/1H2FNeAsKoQqCsO0n7PiA1HcT1ingnwJd/view?usp=sharing)\]

![](https://github.com/alextrevithick/GRF/blob/main/leaves.gif) |  ![](https://github.com/alextrevithick/GRF/blob/main/orchids.gif) | ![](https://github.com/alextrevithick/GRF/blob/main/fortress.gif) |  ![](https://github.com/alextrevithick/GRF/blob/main/trex.gif)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/alextrevithick/GRF/blob/main/room.gif) |  ![](https://github.com/alextrevithick/GRF/blob/main/horns.gif) | ![](https://github.com/alextrevithick/GRF/blob/main/fern.gif) |  ![](https://github.com/alextrevithick/GRF/blob/main/flower.gif)

## Qualitative Results on Synthetic-NeRF and Real Forward-facing
![](https://github.com/alextrevithick/GRF/blob/main/qual_comp_real.png)

## Qualitative Results on Shapenet
![](https://github.com/alextrevithick/GRF/blob/main/car.gif) |  ![](https://github.com/alextrevithick/GRF/blob/main/chair.gif)


## Method
GRF is a powerful implicit neural function that can represent and render arbitrarily complex 3D scenes in a single network only from 2D observations. GRF takes a set of posed 2D images as input, constructs an internal representation for each 3D point of the scene, and renders the corresponding appearance and geometry of any 3D point viewing from an arbitrary angle. The key to our approach is to explicitly integrate the principle of multi-view geometry to obtain features representative of an entire ray from a given viewpoint. Thus, in a single forward pass to render a scene from a novel view, GRF takes some views of that scene as input, computes per-pixel pose-aware features for each ray from the given viewpoints through the image plane at that pixel, and then uses those features to predict the volumetric density and rgb values of points in 3D space. Volumetric rendering is then applied. 

GRF            |  U-Net Design
:-------------------------:|:-------------------------:
![](https://github.com/alextrevithick/GRF/blob/main/fig_GRF.png) |  ![](https://github.com/alextrevithick/GRF/blob/main/fig_U-Net.png)

## Results
Our method achieves extremely high-quality results on three challenging datasets. On the shapenet chairs and cars dataset, GRF outperforms the state of the art on 50-view novel view synthesis.

![](https://github.com/alextrevithick/GRF/blob/main/fig_results_Shapenet.png)

On single-scene reconstruction, our method also outperforms the state of the art on both synthetic datasets and complex real datasets.

![](https://github.com/alextrevithick/GRF/blob/main/fig_results_Syn.png)


![](https://github.com/alextrevithick/GRF/blob/main/fig_results_LLFF.png)




