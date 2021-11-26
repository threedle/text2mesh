---
layout: default
---
<center>

<img src="figures/hero/animals387subdiv.gif" alt="horse with poncho" width="400"/>  <img src="figures/hero/person.gif" alt="batman" width="400"/>

*examples of meshes yielded by _NSF_*

* * *

</center>

> In this work, we develop intuitive controls for editing
> the style of 3D objects. Our framework, Text2Mesh, stylizes a 3D mesh by predicting color and local geometric details which conform to a target text prompt. We consider
> a disentangled representation of a 3D object using a fixed
> mesh input (content) coupled with a learned neural network, which we term neural style field network. In order
> to modify style, we obtain a similarity score between a text
> prompt (describing style) and a stylized mesh by harnessing the representational power of CLIP. Text2Mesh requires
> neither a pre-trained generative model nor a specialized
> 3D mesh dataset. It can handle low-quality meshes (nonmanifold, boundaries, etc.) with arbitrary genus, and does
> not require UV parameterization. We demonstrate the ability of our technique to synthesize a myriad of styles over a
> wide variety of 3D meshes.

![Pipeline](figures/pipeline.svg)
*The pipeline of our method*

## Morphs
_NSF_ is capable of generating small deformations over the same source mesh, based on different targets. Thus, it's very natural to morph different results.

| horse        | vase          | chair |
|:-------------|:------------------|:------|
| <img src="figures/morphs/morph_animals387subdiv.gif" alt="horse" width="200"/> | <img src="figures/morphs/morph_vases25subdiv.gif" alt="vase" width="200"/>  | <img src="figures/morphs/morph_chairs432subdiv.gif" alt="chair" width="200"/>  |
| <img src="figures/morphs/morph_animals387subdiv_init.gif" alt="horse" width="200"/> | <img src="figures/morphs/morph_vases25subdiv_init.gif" alt="vase" width="200"/>    | <img src="figures/morphs/morph_chairs432subdiv_init.gif" alt="chair" width="200"/>    |


## Citation
```
@article{Text2Mesh,
  title={Text2Mesh: Text-Driven Neural Stylization for Meshes},
  author={}
}
```