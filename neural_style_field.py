import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os
from utils import FourierFeatureTransform
from utils import device


class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size, T, d=3, apply=True):
        super(ProgressiveEncoding, self).__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.apply = apply
    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(
            2)  # no need to reduce d or to check cases
        if not self.apply:
            alpha = torch.ones_like(alpha, device=device)  ## this layer means pure ffn without progress.
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha


class NeuralStyleField(nn.Module):
    # Same base then split into two separate modules 
    def __init__(self, sigma, depth, width, encoding, colordepth=2, normdepth=2, normratio=0.1, clamp=None,
                 normclamp=None,niter=6000, input_dim=3, progressive_encoding=True, exclude=0):
        super(NeuralStyleField, self).__init__()
        self.pe = ProgressiveEncoding(mapping_size=width, T=niter, d=input_dim)
        self.clamp = clamp
        self.normclamp = normclamp
        self.normratio = normratio
        layers = []
        if encoding == 'gaussian':
            layers.append(FourierFeatureTransform(input_dim, width, sigma, exclude))
            if progressive_encoding:
                layers.append(self.pe)
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.base = nn.ModuleList(layers)

        # Branches 
        color_layers = []
        for _ in range(colordepth):
            color_layers.append(nn.Linear(width, width))
            color_layers.append(nn.ReLU())
        color_layers.append(nn.Linear(width, 3))
        self.mlp_rgb = nn.ModuleList(color_layers)

        normal_layers = []
        for _ in range(normdepth):
            normal_layers.append(nn.Linear(width, width))
            normal_layers.append(nn.ReLU())
        normal_layers.append(nn.Linear(width, 1))
        self.mlp_normal = nn.ModuleList(normal_layers)

        print(self.base)
        print(self.mlp_rgb)
        print(self.mlp_normal)

    def reset_weights(self):
        self.mlp_rgb[-1].weight.data.zero_()
        self.mlp_rgb[-1].bias.data.zero_()
        self.mlp_normal[-1].weight.data.zero_()
        self.mlp_normal[-1].bias.data.zero_()

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = self.mlp_rgb[0](x)
        for layer in self.mlp_rgb[1:]:
            colors = layer(colors)
        displ = self.mlp_normal[0](x)
        for layer in self.mlp_normal[1:]:
            displ = layer(displ)

        if self.clamp == "tanh":
            colors = F.tanh(colors) / 2
        elif self.clamp == "clamp":
            colors = torch.clamp(colors, 0, 1)
        if self.normclamp == "tanh":
            displ = F.tanh(displ) * self.normratio
        elif self.normclamp == "clamp":
            displ = torch.clamp(displ, -self.normratio, self.normratio)

        return colors, displ



def save_model(model, loss, iter, optim, output_dir):
    save_dict = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss
    }

    path = os.path.join(output_dir, 'checkpoint.pth.tar')

    torch.save(save_dict, path)


