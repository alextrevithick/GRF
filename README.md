# GRF
This is the official repository for General Radiance Field (GRF).\\ 
\[Paper\] \[Code\] \[Video\]

## Method
GRF is a powerful implicit neural function that can represent and render arbitrarily complex 3D scenes in a single network only from 2D observations. GRF takes a set of posed 2D images as input, constructs an internal representation for each 3D point of the scene, and renders the corresponding appearance and geometry of any 3D point viewing from an arbitrary angle. The key to our approach is to explicitly integrate the principle of multi-view geometry to obtain features representative of an entire ray from a given viewpoint. Thus, in a single forward pass to render a scene from a novel view, GRF takes some views of that scene as input, computes per-pixel pose-dependent features for each ray from the given viewpoints through the image plane at that pixel, and then uses those features to predict the volumetric density and rgb values of points in 3D space. Volumetric rendering is then applied.

GRF            |  U-Net Design
:-------------------------:|:-------------------------:
![](https://github.com/alextrevithick/GRF/blob/main/fig_GRF.png) |  ![](https://github.com/alextrevithick/GRF/blob/main/fig_U-Net.png)



