# Text2Mesh
**Text2Mesh** is a method for text-driven stylization of a 3D mesh, as described in "Text2Mesh: Text-Driven Neural Stylization for Meshes" (forthcoming).

## Installation
```
conda env create --file text2mesh.yml
conda activate text2mesh
```

## System Requirements
- Python == 3.7
- CUDA == 10.2.0
- GPU w/ 8 GB ram, CUDA 10.2 compatible

## Run examples
Call the below shell scripts to generate example styles. 
```bash
# batman 
./run_batman.sh
# shoe made of cactus 
./run_shoe.sh
# colorful crochet vase 
./run_vase.sh
```
The outputs will be saved to `results/demo`, with the stylized .obj files, colored and uncolored render views, and screenshots during training.

#### Outputs
<p float="center">
<img alt="person" height="180" src="images/person.png" width="320"/>
<img alt="batman geometry" height="180" src="images/batman_init.png" width="320"/>
<img alt="batman style" height="180" src="images/batman_final.png" width="320"/>
</p>

<p float="center">
<img alt="shoe" height="180" src="images/shoe.png" width="320"/>
<img alt="shoe geometry" height="180" src="images/shoe_init.png" width="320"/>
<img alt="shoe style" height="180" src="images/shoe_final.png" width="320"/>
</p>

<p float="center">
<img alt="vase" height="180" src="images/vase.png" width="320"/>
<img alt="vase geometry" height="180" src="images/vase_init.png" width="320"/>
<img alt="vase style" height="180" src="images/vase_final.png" width="320"/>
</p>

## Citation
```
@article{text2mesh,
    author = {Michel, Oscar
              and Baron-On, Roi
              and Liu, Richard
              and Benaim, Sagie
              and Hanocka, Rana
              },
    title = {{Text2Mesh: Text-Driven Neural Stylization for Meshes}},
    journal = {TODO: ARXIV},
    year  = {2021}
}
```