import copy
from . import Normalizer


class MeshNormalizer:
    def __init__(self, mesh):
        self._mesh = mesh  # original copy of the mesh
        self.normalizer = Normalizer.get_bounding_sphere_normalizer(self._mesh.vertices)

    def __call__(self):
        self._mesh.vertices = self.normalizer(self._mesh.vertices)
        return self._mesh

