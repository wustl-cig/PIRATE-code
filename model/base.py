import torch
from torch import nn
import torch.nn.functional as F

class DnCNN(nn.Module):
    def __init__(self, depth=5, in_channels=3, out_channels=3, init_features=64, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        
        layers.append(nn.Conv3d(
            in_channels=in_channels, out_channels=init_features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ELU())

        for _ in range(depth-1):
            layers.append((nn.Conv3d(
                in_channels=init_features, out_channels=init_features, kernel_size=kernel_size, padding=padding, bias=True)))
            layers.append(nn.ELU())

        layers.append(nn.Conv3d(
            in_channels=init_features, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True))

        self.dncnn_3 = nn.Sequential(*layers)

    def forward(self, x):
        out =  self.dncnn_3(x)
        return out
    
class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, return_phi=False):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if return_phi:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode), new_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        
class ResizeTransform(nn.Module):

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x
    

def anderson_solver(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0):
    """ Anderson's acceleration for fixed point iteration. """

    bsz, C, H, W, D  = x0.shape
    X = torch.zeros(bsz, m, C * H * W * D, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, C * H * W * D, dtype=x0.dtype, device=x0.device)

    # bsz, d, H, W = x0.shape
    # X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    # F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)

    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0, 0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0), 1).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []

    iter_ = range(2, max_iter)

    for k in iter_:
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
#         alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)
#         print(H.shape, y.shape)
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
#         print(alpha.shape)
#         alpha = alpha[0][:, 1:n + 1, 0]
        
        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0), k).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        if res[-1] < tol:
            break

    return X[:, k % m].view_as(x0), res



