import pymeshlab 
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", type=str)
parser.add_argument("--output_path", type=str, default="./remeshed_obj.obj")

args = parser.parse_args()

ms = pymeshlab.MeshSet()

ms.load_new_mesh(args.obj_path)
ms.meshing_isotropic_explicit_remeshing()

ms.save_current_mesh(args.output_path)