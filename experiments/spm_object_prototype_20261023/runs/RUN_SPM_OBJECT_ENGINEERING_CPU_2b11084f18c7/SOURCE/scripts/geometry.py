"""Known-camera projection and deterministic hard triangle rasterization."""
import math
import torch


def camera(azimuth, elevation, distance=2.0, focal=170.0):
    az, el = math.radians(azimuth), math.radians(elevation)
    eye = torch.tensor([math.cos(az)*math.cos(el), math.sin(az)*math.cos(el), math.sin(el)])*distance
    forward = -eye/eye.norm()
    right = torch.linalg.cross(forward, torch.tensor([0., 0., 1.]))
    right /= right.norm()
    up = torch.linalg.cross(right, forward)
    rotation = torch.stack([right, up, forward])
    return {'matrix': torch.cat([rotation, (-rotation@eye)[:, None]], 1), 'focal': focal}


def project(vertices, matrix, focal, size):
    # matrix can be 3x4 or Bx3x4; vertices can be Vx3 or BxVx3.
    points = vertices @ matrix[..., :3].transpose(-1, -2) + matrix[..., 3].unsqueeze(-2)
    uv = points[..., :2]/points[..., 2:].clamp_min(.05)*focal
    uv = uv*vertices.new_tensor([1., -1.])+size/2
    return uv, points[..., 2]


@torch.no_grad()
def raster(vertices, faces, matrix, focal, size, attributes=None):
    """Pixel centers, perspective-correct barycentrics, two-sided surfaces, nearest depth."""
    uv, depth = project(vertices, matrix, focal, size)
    xy = uv[faces]
    zz = depth[faces]
    yy, xx = torch.meshgrid(torch.arange(size, device=vertices.device)+.5,
                            torch.arange(size, device=vertices.device)+.5, indexing='ij')
    x, y = xx.flatten()[None], yy.flatten()[None]
    a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
    den = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0])+(c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    valid_triangle = (den.abs() > 1e-8) & (zz.min(1).values > .05)
    safe = torch.where(valid_triangle, den, torch.ones_like(den))[:, None]
    w0 = ((b[:, 1:2]-c[:, 1:2])*(x-c[:, 0:1])+(c[:, 0:1]-b[:, 0:1])*(y-c[:, 1:2]))/safe
    w1 = ((c[:, 1:2]-a[:, 1:2])*(x-c[:, 0:1])+(a[:, 0:1]-c[:, 0:1])*(y-c[:, 1:2]))/safe
    w2 = 1-w0-w1
    valid = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7) & valid_triangle[:, None]
    inverse = w0/zz[:, 0:1].clamp_min(.05)+w1/zz[:, 1:2].clamp_min(.05)+w2/zz[:, 2:3].clamp_min(.05)
    z = torch.where(valid, 1/inverse.clamp_min(1e-8), torch.inf)
    nearest, indices = z.min(0)
    visible = torch.isfinite(nearest)
    pix = torch.arange(size*size, device=vertices.device)
    perspective = torch.stack([w0[indices, pix], w1[indices, pix], w2[indices, pix]], 1)
    perspective = perspective/zz[indices].clamp_min(.05)
    perspective /= perspective.sum(1, keepdim=True).clamp_min(1e-8)
    perspective[~visible] = 0
    values = {}
    for key, value in (attributes or {}).items():
        values[key] = (value[faces[indices]]*perspective[:, :, None]).sum(1).reshape(size, size, -1)
    tri = indices.reshape(size, size).clone()
    tri[~visible.reshape(size, size)] = -1
    return dict(mask=visible.reshape(size, size), depth=torch.where(visible, nearest, 0).reshape(size, size),
                triangle=tri, barycentric=perspective.reshape(size, size, 3), attributes=values)


def skin(canonical, transforms, weights):
    # canonical B,V,3; transforms B,J,3,4; weights V,J.
    homogeneous = torch.cat([canonical, torch.ones_like(canonical[..., :1])], -1)
    return torch.einsum('vj,bjkl,bvl->bvk', weights, transforms, homogeneous)


def foreground_bar(matrix):
    # A box in camera coordinates, closer than the animal; transformed to world coordinates.
    cam_vertices = torch.tensor([[x, y, z] for x in [-.045, .045] for y in [-.55, .55] for z in [.95, 1.02]],
                                device=matrix.device)
    triangles = torch.tensor([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                              [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]], device=matrix.device)
    vertices = (cam_vertices-matrix[:, 3])@matrix[:, :3]
    return vertices, triangles
