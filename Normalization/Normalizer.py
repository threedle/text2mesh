import torch


class Normalizer:
    @classmethod
    def get_bounding_box_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=1, dim=1))
        return Normalizer(scale=scale, shift=shift)

    @classmethod
    def get_bounding_sphere_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=2, dim=1))
        return Normalizer(scale=scale, shift=shift)

    def __init__(self, scale, shift):
        self._scale = scale
        self._shift = shift

    def __call__(self, x):
        return (x-self._shift) / self._scale

    def get_de_normalizer(self):
        inv_scale = 1 / self._scale
        inv_shift = -self._shift / self._scale
        return Normalizer(scale=inv_scale, shift=inv_shift)