---
layout: default
---

<center>

<img src="figures/teaser/ironman_crop.gif" alt="ironman" width="150"/> <img src="figures/teaser/lamp_crop.gif" alt="a lamp made of brick" width="250"/> <img src="figures/teaser/candle_crop.gif" alt="a candle made of colorful crochet" width="150"/> <img src="figures/teaser/horse_crop.gif" alt="a horse wearing an astronaut suit" width="200"/>
<p><em>
Text2Mesh produces color and geometric details over a variety of source meshes, driven by a target text prompt. Our stylization results coherently blend unique and ostensibly unrelated combinations of text, capturing both global semantics and part-aware attributes.
</em></p>

<a href="https://github.com/threedle/text2mesh" class="btn">Code</a>
<a href="https://www.dropbox.com/s/hbzq7xp95tr1788/text2mesh-preprint.pdf?dl=0" class="btn">Paper</a>
<a href="arxiv.com/supp" class="btn">Supplementary</a>

</center>

* * *

## Abstract

In this work, we develop intuitive controls for editing the style of 3D objects. Our framework, Text2Mesh, stylizes a 3D mesh by predicting color and local geometric details which conform to a target text prompt. We consider a disentangled representation of a 3D object using a fixed mesh input (content) coupled with a learned neural network, which we term neural style field network. In order to modify style, we obtain a similarity score between a text prompt (describing style) and a stylized mesh by harnessing the representational power of CLIP. Text2Mesh requires neither a pre-trained  generative model nor a specialized 3D mesh dataset. It can handle low-quality meshes (non-manifold, boundaries, etc.) with arbitrary genus, and does not require UV parameterization. We demonstrate the ability of our technique to synthesize a myriad of styles over a wide variety of 3D meshes.

## Overview
<center>
<img src="figures/pipeline.jpg" alt="Pipeline" width="1000"/>
<p><em>Text2Mesh modifies an <span style="color: palegreen">input mesh</span> to conform to the <span style="color: palegreen">target text</span> by predicting color and geometric details. The weights of the <span style="color: sandybrown">neural style network</span> are optimized by <span style="color: royalblue">rendering</span> multiple 2D images and applying <span style="color: royalblue">2D augmentations</span>, which are given a similarity score to the target from the CLIP-based <span style="color: salmon">semantic loss</span>.</em></p>
</center>

## View Consistency
We use [CLIP's](https://openai.com/blog/clip/) ability to jointly embed text and images to produce view-consistent and semantically meaningful stylizations over the entire 3D shape.
<center>
<div class="carousel">
    <div class="item">
        <img src="figures/multiple-views/croissant_final_crop.gif" alt="croissant made of colorful crochet" width="500"/>
        <p>a croissant made of colorful crochet</p>
    </div>
    <div class="item">
        <img src="figures/multiple-views/armadillo_final_crop.gif" alt="armadillo made of gold" width="500"/>
        <p>armadillo made of gold</p>
    </div>
    <div class="item">
        <img src="figures/multiple-views/donkey_final_crop.gif" alt="donkey wearing jeans" width="500"/>
        <p>donkey wearing jeans</p>
    </div>
</div>
</center>
## General Stylization
For the same input mesh, Text2Mesh is capable of generating a variety of different local geometric displacements to synthesize a wide range of styles.
<center>
<div class="double-carousel">
    <div class="item">
        <img src="figures/morphs/vase_init_crop.gif" alt="vase" width="400"/>
        <img src="figures/morphs/vase_full_crop.gif" alt="vase" width="400"/>
    </div>
    <div class="item">
        <img src="figures/morphs/donut_init_crop.gif" alt="donut" width="400"/>
        <img src="figures/morphs/donut_full_crop.gif" alt="donut" width="400"/>
    </div>
    <div class="item">
        <img src="figures/morphs/camel_init_crop.gif" alt="camel" width="400"/>
        <img src="figures/morphs/camel_full_crop.gif" alt="camel" width="400"/>
    </div>
    <div class="item">
        <img src="figures/morphs/chair_init_crop.gif" alt="chair" width="400"/>
        <img src="figures/morphs/chair_full_crop.gif" alt="chair" width="400"/>
    </div>
    <div class="item">
        <img src="figures/morphs/alien_init_crop.gif" alt="alien" width="400"/>
        <img src="figures/morphs/alien_full_crop.gif" alt="alien" width="400"/>
    </div>
 </div>
</center>

[comment]: <> (## Interplay of Geometry and Color)

[comment]: <> (We observe a strong correlation between the displacements and the coloring that _NSF_ produces, which results in a consistent stylized 3D mesh.)

[comment]: <> (<center>)

[comment]: <> ( <img src="figures/coupling/donut.gif" alt="a donut with sprinkles" width="600"/>)

[comment]: <> (</center>)

## Beyond Text-Driven Manipulation
We further leverage the joint vision-language embedding space to demonstrate the multi-modal stylization capabilities of our method.

### Image and Mesh Targets
<center>
<div class="carousel">
    <div class="item">
        <img src="figures/target-image/bucket_cobble.gif" alt="bucket" width="340"/>
        <p>image target</p>
    </div>
    <div class="item">
        <img src="figures/target-image/pig_fish.gif" alt="pig" width="500"/>
        <p>image target</p>
    </div>
    <div class="item">
        <img src="figures/target-image/iron_crochet.gif" alt="iron" width="450"/>
        <p>image target</p>
    </div>
    <div class="item">
        <img src="figures/target-mesh/armadillo_final.gif" alt="armadillo" width="400"/>
        <p>mesh target</p>
    </div>
 
 </div>
</center>

## Citation

```
@article{text2mesh,
    author = {Michel, Oscar
              and Bar-On, Roi
              and Liu, Richard
              and Benaim, Sagie
              and Hanocka, Rana
              },
    title = {{Text2Mesh: Text-Driven Neural Stylization for Meshes}},
    journal = {TODO: ARXIV},
    year  = {2021}
}
```