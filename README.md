# Text2Mesh [[Project Page](https://github.com/threedle/text2mesh)]
[![arXiv](https://img.shields.io/badge/arXiv-Text2Mesh-b31b1b.svg)](https://arxiv.org/abs/1234.56789)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.9.0-Red?logo=pytorch)
![crochet candle](images/vases.gif)
**Text2Mesh** is a method for text-driven stylization of a 3D mesh, as described in "Text2Mesh: Text-Driven Neural Stylization for Meshes" (forthcoming).

## Getting Started
### Installation

**Note:** The below installation will fail if run on something other than a CUDA GPU machine.
```
conda env create --file text2mesh.yml
conda activate text2mesh
```
<details>
<summary>System requirements <em>[click to expand]</em> </summary>
### System Requirements
- Python 3.7
- CUDA 10.2
- GPU w/ minimum 8 GB ram
</details>

### Run examples
Call the below shell scripts to generate example styles. 
```bash
# steve jobs 
./demo/run_jobs.sh
# shoe made of cactus 
./demo/run_shoe.sh
# colorful crochet vase 
./demo/run_vase.sh
```
The outputs will be saved to `results/demo`, with the stylized .obj files, colored and uncolored render views, and screenshots during training.

#### Outputs
<p float="center">
<img alt="alien" height="135" src="images/alien.png" width="240"/>
<img alt="alien geometry" height="135" src="images/alien_cobble_init.png" width="240"/>
<img alt="alien style" height="135" src="images/alien_cobble_final.png" width="240"/>
</p>

<p float="center">
<img alt="alien" height="135" src="images/alien.png" width="240"/>
<img alt="alien geometry" height="135" src="images/alien_wood_init.png" width="240"/>
<img alt="alien style" height="135" src="images/alien_wood_final.png" width="240"/>
</p>

<p float="center">
<img alt="candle" height="135" src="images/candle.png" width="240"/>
<img alt="candle geometry" height="135" src="images/candle_init.png" width="240"/>
<img alt="candle style" height="135" src="images/candle_final.png" width="240"/>
</p>

<p float="center">
<img alt="person" height="135" src="images/person.png" width="240"/>
<img alt="ninja geometry" height="135" src="images/ninja_init.png" width="240"/>
<img alt="ninja style" height="135" src="images/ninja_final.png" width="240"/>
</p>

<p float="center">
<img alt="shoe" height="135" src="images/shoe.png" width="240"/>
<img alt="shoe geometry" height="135" src="images/shoe_init.png" width="240"/>
<img alt="shoe style" height="135" src="images/shoe_final.png" width="240"/>
</p>

<p float="center">
<img alt="vase" height="135" src="images/vase.png" width="240"/>
<img alt="vase geometry" height="135" src="images/vase_init.png" width="240"/>
<img alt="vase style" height="135" src="images/vase_final.png" width="240"/>
</p>

<p float="center">
<img alt="lamp" height="135" src="images/lamp.png" width="240"/>
<img alt="lamp geometry" height="135" src="images/lamp_init.png" width="240"/>
<img alt="lamp style" height="135" src="images/lamp_final.png" width="240"/>
</p>

<p float="center">
<img alt="horse" height="135" src="images/horse.png" width="240"/>
<img alt="horse geometry" height="135" src="images/horse_init.png" width="240"/>
<img alt="horse style" height="135" src="images/horse_final.png" width="240"/>
</p>

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
