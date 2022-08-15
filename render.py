from mesh import Mesh
import kaolin as kal
from utils import get_camera_from_view2
import matplotlib.pyplot as plt
from utils import device
import torch
import numpy as np


class Renderer():

    def __init__(self, mesh='sample.obj',
                 lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
                 dim=(224, 224)):

        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim

    def render_y_views(self, mesh, num_views=8, show=False, lighting=True, background=None, mask=False):

        faces = mesh.faces
        n_faces = faces.shape[0]

        azim = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]  # since 0 =360 dont include last element
        # elev = torch.cat((torch.linspace(0, np.pi/2, int((num_views+1)/2)), torch.linspace(0, -np.pi/2, int((num_views)/2))))
        elev = torch.zeros(len(azim))
        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(images[i].permute(1,2,0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def render_single_view(self, mesh, elev=0, azim=0, show=False, lighting=True, background=None, radius=2,
                           return_mask=False):
        # if mesh is None:
        #     mesh = self._current_mesh
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        camera_transform = get_camera_from_view2(torch.tensor(elev), torch.tensor(azim), r=radius).to(device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection, camera_transform=camera_transform)

        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # Debugging: color where soft mask is 1
        # tmp_rgb = torch.ones((224,224,3))
        # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
        # rgb_mask.append(tmp_rgb)

        if background is not None:
            image_features, mask = image_features

        image = torch.clamp(image_features, 0.0, 1.0)

        if lighting:
            image_normals = face_normals[:, face_idx].squeeze(0)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
            image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
            image = torch.clamp(image, 0.0, 1.0)

        if background is not None:
            background_mask = torch.zeros(image.shape).to(device)
            mask = mask.squeeze(-1)
            assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
            background_mask[torch.where(mask == 0)] = background
            image = torch.clamp(image + background_mask, 0., 1.)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(figsize=(89.6, 22.4))
                axs.imshow(image[0].cpu().numpy())
                # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if return_mask == True:
            return image.permute(0, 3, 1, 2), mask
        return image.permute(0, 3, 1, 2)

    def render_uniform_views(self, mesh, num_views=8, show=False, lighting=True, background=None, mask=False,
                             center=[0, 0], radius=2.0):

        # if mesh is None:
        #     mesh = self._current_mesh

        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[
               :-1]  # since 0 =360 dont include last element
        elev = torch.cat((torch.linspace(center[1], np.pi / 2 + center[1], int((num_views + 1) / 2)),
                          torch.linspace(center[1], -np.pi / 2 + center[1], int((num_views) / 2))))
        images = []
        masks = []
        background_masks = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            # tmp_rgb = torch.ones((224,224,3))
            # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
            # rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                background_masks.append(background_mask)
                image = torch.clamp(image + background_mask, 0., 1.)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        if background is not None:
            background_masks = torch.cat(background_masks, dim=0).permute(0, 3, 1, 2)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(background_masks[i].permute(1,2,0).cpu().numpy())
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def render_front_views(self, mesh, num_views=8, std=8, center_elev=0, center_azim=0, show=False, lighting=True,
                           background=None, mask=False, return_views=False):
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / std + center_elev))
        azim = torch.cat((torch.tensor([center_azim]), torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))
        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            tmp_rgb = torch.ones((224, 224, 3))
            tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1, 0, 0]).float()
            rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        rgb_mask = torch.cat(rgb_mask, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images

    def render_prompt_views(self, mesh, prompt_views, center=[0, 0], background=None, show=False, lighting=True,
                            mask=False):

        # if mesh is None:
        #     mesh = self._current_mesh

        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]
        num_views = len(prompt_views)

        images = []
        masks = []
        rgb_mask = []
        face_attributes = mesh.face_attributes

        for i in range(num_views):
            view = prompt_views[i]
            if view == "front":
                elev = 0 + center[1]
                azim = 0 + center[0]
            if view == "right":
                elev = 0 + center[1]
                azim = np.pi / 2 + center[0]
            if view == "back":
                elev = 0 + center[1]
                azim = np.pi + center[0]
            if view == "left":
                elev = 0 + center[1]
                azim = 3 * np.pi / 2 + center[0]
            if view == "top":
                elev = np.pi / 2 + center[1]
                azim = 0 + center[0]
            if view == "bottom":
                elev = -np.pi / 2 + center[1]
                azim = 0 + center[0]

            if background is not None:
                face_attributes = [
                    mesh.face_attributes,
                    torch.ones((1, n_faces, 3, 1), device=device)
                ]
            else:
                face_attributes = mesh.face_attributes

            camera_transform = get_camera_from_view2(torch.tensor(elev), torch.tensor(azim), r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if not mask:
            return images
        else:
            return images, masks


if __name__ == '__main__':
    mesh = Mesh('sample.obj')
    mesh.set_image_texture('sample_texture.png')
    renderer = Renderer()
    # renderer.render_uniform_views(mesh,show=True,texture=True)
    mesh = mesh.divide()
    renderer.render_uniform_views(mesh, show=True, texture=True)
