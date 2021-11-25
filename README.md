# Text2Mesh Demo Code

## Installation
```
conda env create --file text2mesh.yml
conda activate text2mesh
```

## System Requirements
- Python == 3.7
- CUDA == 10.2.0
- GPU w/ 8 GB ram, CUDA 10.2 compatible

## Run example
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


