from torch import nn
import numpy as np
import torch.nn.functional as F
from .base import *
from .loss import *

class PIRATE(nn.Module):

    def __init__(self, dnn: nn.Module, config:dict, device):

        super(PIRATE, self).__init__()
        self.dnn = dnn
        self.gamma = torch.tensor(config['gamma_inti'], dtype=torch.float32)
        self.tau = torch.tensor(config['tau_inti'], dtype=torch.float32)
        self.inshape = config['image_shape']
        self.device = device
        self.resize = ResizeTransform(1/2, 3)
        self.max_iter = config['iteration']
        self.weight_grad = config['weight_grad']
        
    def get_grad(self, field, in_image, fixed, device):
        transformer = SpatialTransformer(self.inshape)
        transformer = transformer.to(device)
        image = in_image.clone().detach()
        no_grad_field = field.clone().detach()
        no_grad_field.requires_grad=True
        no_grad_field_full = self.resize(no_grad_field)
        image_pred = transformer(image, no_grad_field_full, return_phi=False)
        loss_func = GCC()
        grad_func = Grad('l2', loss_mult=2).loss
        loss = loss_func(image_pred,fixed) + self.weight_grad*grad_func(no_grad_field)
        loss.backward()
        soft_dc = no_grad_field.grad
        return soft_dc

    def forward(self, field, moving, fixed, interation, flag):

        gamma_recent = 0.5*(1+np.cos(interation*np.pi/self.max_iter)) * self.gamma
    
        if flag == "forward":
            torch.set_grad_enabled(True)
            delta_g = self.get_grad(field, moving, fixed, self.device)
            torch.set_grad_enabled(False)
            
        if flag == "backward":
            delta_g = self.get_grad(field, moving, fixed, self.device)

        xSubD = self.tau * self.dnn(field)
        
        xnext  =  field - gamma_recent * (delta_g.detach() + xSubD)

        return xnext