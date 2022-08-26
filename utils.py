import torch
import kaolin as kal
import clip
import numpy as np
from torchvision import transforms
from pathlib import Path

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_camera_from_view(elev, azim, r=3.0):
    x = r * torch.cos(azim) * torch.sin(elev)
    y = r * torch.sin(azim) * torch.sin(elev)
    z = r * torch.cos(elev)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj


def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj


def get_homogenous_coordinates(V):
    N, D = V.shape
    bottom = torch.ones(N, device=device).unsqueeze(1)
    return torch.cat([V, bottom], dim=1)


def apply_affine(verts, A):
    verts = verts.to(device)
    verts = get_homogenous_coordinates(verts)
    A = torch.cat([A, torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0)], dim=0)
    transformed_verts = A @ verts.T
    transformed_verts = transformed_verts[:-1]
    return transformed_verts.T

def standardize_mesh(mesh):
    verts = mesh.vertices
    center = verts.mean(dim=0)
    verts -= center
    scale = torch.std(torch.norm(verts, p=2, dim=1))
    verts /= scale
    mesh.vertices = verts
    return mesh


def normalize_mesh(mesh):
    verts = mesh.vertices

    # Compute center of bounding box
    # center = torch.mean(torch.column_stack([torch.max(verts, dim=0)[0], torch.min(verts, dim=0)[0]]))
    center = verts.mean(dim=0)
    verts = verts - center
    scale = torch.max(torch.norm(verts, p=2, dim=1))
    verts = verts / scale
    mesh.vertices = verts
    return mesh


def get_texture_map_from_color(mesh, color, H=224, W=224):
    num_faces = mesh.faces.shape[0]
    texture_map = torch.zeros(1, H, W, 3).to(device)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)
    face_attributes[:, :, :] = color
    return face_attributes


def sample_bary(faces, vertices):
    num_faces = faces.shape[0]
    num_vertices = vertices.shape[0]

    # get random barycentric for each face TODO: improve sampling
    A = torch.randn(num_faces)
    B = torch.randn(num_faces) * (1 - A)
    C = 1 - (A + B)
    bary = torch.vstack([A, B, C]).to(device)

    # compute xyz of new vertices and new uvs (if mesh has them)
    new_vertices = torch.zeros(num_faces, 3).to(device)
    new_uvs = torch.zeros(num_faces, 2).to(device)
    face_verts = kal.ops.mesh.index_vertices_by_faces(vertices.unsqueeze(0), faces)
    for f in range(num_faces):
        new_vertices[f] = bary[:, f] @ face_verts[:, f]
    new_vertices = torch.cat([vertices, new_vertices])
    return new_vertices


def add_vertices(mesh):
    faces = mesh.faces
    vertices = mesh.vertices
    num_faces = faces.shape[0]
    num_vertices = vertices.shape[0]

    # get random barycentric for each face TODO: improve sampling
    A = torch.randn(num_faces)
    B = torch.randn(num_faces) * (1 - A)
    C = 1 - (A + B)
    bary = torch.vstack([A, B, C]).to(device)

    # compute xyz of new vertices and new uvs (if mesh has them)
    new_vertices = torch.zeros(num_faces, 3).to(device)
    new_uvs = torch.zeros(num_faces, 2).to(device)
    face_verts = kal.ops.mesh.index_vertices_by_faces(vertices.unsqueeze(0), faces)
    face_uvs = mesh.face_uvs
    for f in range(num_faces):
        new_vertices[f] = bary[:, f] @ face_verts[:, f]
        if face_uvs is not None:
            new_uvs[f] = bary[:, f] @ face_uvs[:, f]

    # update face and face_uvs of mesh
    new_vertices = torch.cat([vertices, new_vertices])
    new_faces = []
    new_face_uvs = []
    new_vertex_normals = []
    for i in range(num_faces):
        old_face = faces[i]
        a, b, c = old_face[0], old_face[1], old_face[2]
        d = num_vertices + i
        new_faces.append(torch.tensor([a, b, d]).to(device))
        new_faces.append(torch.tensor([a, d, c]).to(device))
        new_faces.append(torch.tensor([d, b, c]).to(device))
        if face_uvs is not None:
            old_face_uvs = face_uvs[0, i]
            a, b, c = old_face_uvs[0], old_face_uvs[1], old_face_uvs[2]
            d = new_uvs[i]
            new_face_uvs.append(torch.vstack([a, b, d]))
            new_face_uvs.append(torch.vstack([a, d, c]))
            new_face_uvs.append(torch.vstack([d, b, c]))
        if mesh.face_normals is not None:
            new_vertex_normals.append(mesh.face_normals[i])
        else:
            e1 = vertices[b] - vertices[a]
            e2 = vertices[c] - vertices[a]
            norm = torch.cross(e1, e2)
            norm /= torch.norm(norm)

            # Double check sign against existing vertex normals
            if torch.dot(norm, mesh.vertex_normals[a]) < 0:
                norm = -norm

            new_vertex_normals.append(norm)

    vertex_normals = torch.cat([mesh.vertex_normals, torch.stack(new_vertex_normals)])

    if face_uvs is not None:
        new_face_uvs = torch.vstack(new_face_uvs).unsqueeze(0).view(1, 3 * num_faces, 3, 2)
    new_faces = torch.vstack(new_faces)

    return new_vertices, new_faces, vertex_normals, new_face_uvs


def get_rgb_per_vertex(vertices, faces, face_rgbs):
    num_vertex = vertices.shape[0]
    num_faces = faces.shape[0]
    vertex_color = torch.zeros(num_vertex, 3)

    for v in range(num_vertex):
        for f in range(num_faces):
            face = num_faces[f]
            if v in face:
                vertex_color[v] = face_rgbs[f]
    return face_rgbs


def get_barycentric(p, faces):
    # faces num_points x 3 x 3
    # p num_points x 3
    # source: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates

    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]

    v0, v1, v2 = b - a, c - a, p - a
    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - (w + v)

    return torch.vstack([u, v, w]).T


def get_uv_assignment(num_faces):
    M = int(np.ceil(np.sqrt(num_faces)))
    uv_map = torch.zeros(1, num_faces, 3, 2).to(device)
    px, py = 0, 0
    count = 0
    for i in range(M):
        px = 0
        for j in range(M):
            uv_map[:, count] = torch.tensor([[px, py],
                                             [px + 1, py],
                                             [px + 1, py + 1]])
            px += 2
            count += 1
            if count >= num_faces:
                hw = torch.max(uv_map.view(-1, 2), dim=0)[0]
                uv_map = (uv_map - hw / 2.0) / (hw / 2)
                return uv_map
        py += 2


def get_texture_visual(res, nt, mesh):
    faces_vt = kal.ops.mesh.index_vertices_by_faces(mesh.vertices.unsqueeze(0), mesh.faces).squeeze(0)

    # as to not include encpoint, gen res+1 points and take first res
    uv = torch.cartesian_prod(torch.linspace(-1, 1, res + 1)[:-1], torch.linspace(-1, 1, res + 1))[:-1].to(device)
    image = torch.zeros(res, res, 3).to(device)
    # image[:,:,:] = torch.tensor([0.0,1.0,0.0]).to(device)
    image = image.permute(2, 0, 1)
    num_faces = mesh.faces.shape[0]
    uv_map = get_uv_assignment(num_faces).squeeze(0)

    zero = torch.tensor([0.0, 0.0, 0.0]).to(device)
    one = torch.tensor([1.0, 1.0, 1.0]).to(device)

    for face in range(num_faces):
        bary = get_barycentric(uv, uv_map[face].repeat(len(uv), 1, 1))

        maskA = torch.logical_and(bary[:, 0] >= 0.0, bary[:, 0] <= 1.0)
        maskB = torch.logical_and(bary[:, 1] >= 0.0, bary[:, 1] <= 1.0)
        maskC = torch.logical_and(bary[:, 2] >= 0.0, bary[:, 2] <= 1.0)

        mask = torch.logical_and(maskA, maskB)
        mask = torch.logical_and(maskC, mask)

        inside_triangle = bary[mask]
        inside_triangle_uv = inside_triangle @ uv_map[face]
        inside_triangle_xyz = inside_triangle @ faces_vt[face]
        inside_triangle_rgb = nt(inside_triangle_xyz)

        pixels = (inside_triangle_uv + 1.0) / 2.0
        pixels = pixels * res
        pixels = torch.floor(pixels).type(torch.int64)

        image[:, pixels[:, 0], pixels[:, 1]] = inside_triangle_rgb.T

    return image


# Get rotation matrix about vector through origin
def getRotMat(axis, theta):
    """
    axis: np.array, normalized vector
    theta: radians
    """
    import math

    axis = axis / np.linalg.norm(axis)
    cprod = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rot = math.cos(theta) * np.identity(3) + math.sin(theta) * cprod + \
          (1 - math.cos(theta)) * np.outer(axis, axis)
    return rot


# Map vertices and subset of faces to 0-indexed vertices, keeping only relevant vertices
def trimMesh(vertices, faces):
    unique_v = np.sort(np.unique(faces.flatten()))
    v_val = np.arange(len(unique_v))
    v_map = dict(zip(unique_v, v_val))
    new_faces = np.array([v_map[i] for i in faces.flatten()]).reshape(faces.shape[0], faces.shape[1])
    new_v = vertices[unique_v]

    return new_v, new_faces


# ================== VISUALIZATION =======================
# Back out camera parameters from view transform matrix
def extract_from_gl_viewmat(gl_mat):
    gl_mat = gl_mat.reshape(4, 4)
    s = gl_mat[0, :3]
    u = gl_mat[1, :3]
    f = -1 * gl_mat[2, :3]
    coord = gl_mat[:3, 3]  # first 3 entries of the last column
    camera_location = np.array([-s, -u, f]).T @ coord
    target = camera_location + f * 10  # any scale
    return camera_location, target


def psScreenshot(vertices, faces, axis, angles, save_path, name="mesh", frame_folder="frames", scalars=None,
                 colors=None,
                 defined_on="faces", highlight_faces=None, highlight_color=[1, 0, 0], highlight_radius=None,
                 cmap=None, sminmax=None, cpos=None, clook=None, save_video=False, save_base=False,
                 ground_plane="tile_reflection", debug=False, edge_color=[0, 0, 0], edge_width=1, material=None):
    import polyscope as ps

    ps.init()
    # Set camera to look at same fixed position in centroid of original mesh
    # center = np.mean(vertices, axis = 0)
    # pos = center + np.array([0, 0, 3])
    # ps.look_at(pos, center)
    ps.set_ground_plane_mode(ground_plane)

    frame_path = f"{save_path}/{frame_folder}"
    if save_base == True:
        ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, enabled=True,
                                           edge_color=edge_color, edge_width=edge_width, material=material)
        ps.screenshot(f"{frame_path}/{name}.png")
        ps.remove_all_structures()
    Path(frame_path).mkdir(parents=True, exist_ok=True)
    # Convert 2D to 3D by appending Z-axis
    if vertices.shape[1] == 2:
        vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=1)

    for i in range(len(angles)):
        rot = getRotMat(axis, angles[i])
        rot_verts = np.transpose(rot @ np.transpose(vertices))

        ps_mesh = ps.register_surface_mesh("mesh", rot_verts, faces, enabled=True,
                                           edge_color=edge_color, edge_width=edge_width, material=material)
        if scalars is not None:
            ps_mesh.add_scalar_quantity(f"scalar", scalars, defined_on=defined_on,
                                        cmap=cmap, enabled=True, vminmax=sminmax)
        if colors is not None:
            ps_mesh.add_color_quantity(f"color", colors, defined_on=defined_on,
                                       enabled=True)
        if highlight_faces is not None:
            # Create curve to highlight faces
            curve_v, new_f = trimMesh(rot_verts, faces[highlight_faces, :])
            curve_edges = []
            for face in new_f:
                curve_edges.extend(
                    [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]])
            curve_edges = np.array(curve_edges)
            ps_curve = ps.register_curve_network("curve", curve_v, curve_edges, color=highlight_color,
                                                 radius=highlight_radius)

        if cpos is None or clook is None:
            ps.reset_camera_to_home_view()
        else:
            ps.look_at(cpos, clook)

        if debug == True:
            ps.show()
        ps.screenshot(f"{frame_path}/{name}_{i}.png")
        ps.remove_all_structures()
    if save_video == True:
        import glob
        from PIL import Image
        fp_in = f"{frame_path}/{name}_*.png"
        fp_out = f"{save_path}/{name}.gif"
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)


# ================== POSITIONAL ENCODERS =============================
class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)  # for sape

    def forward(self, x):
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        # x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        res = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        # x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        # x = x.permute(0, 3, 1, 2)

        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)
