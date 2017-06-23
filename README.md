# Spatial Transformer Networks

[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)  (STN) is a dynamic mechanism that produces transformations of input images (or feature maps)including  scaling, cropping, rotations, as well as non-rigid deformations. This enables the network to not only select regions of an image that are most relevant (attention), but also to transform those regions to simplify recognition in the following layers.

In this repositary, we implemented a STN for [2D Affine Transformation](https://en.wikipedia.org/wiki/Affine_transformation) on MNIST dataset. We generated images with size of 40x40 from the original MNIST dataset, and distorted the images by random rotation, shifting, shearing and zoom in/out. The STN was able to learn to automatically apply transformations on distorted images via classification task.

<table class="image">
<div align="center">
    <img src="https://github.com/zsdonghao/tl-book/blob/master/images/cifar-10.jpg?raw=true"/>  
    <br>  
    <em align="center">Image Caption is here~~</em>  
</div>
</table>