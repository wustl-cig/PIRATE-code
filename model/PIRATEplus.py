from torch import nn
import numpy as np
import torch.nn.functional as F
from .base import *
from .loss import *
from .PIRATE import *

class PIRATEplus(nn.Module):

    def __init__(self, PIRATE, config):
        super().__init__()

        self.PIRATE = PIRATE
        self.max_iter = config["max_iter"]
        self.tol = config["tol"]

    def forward(self, field, moving, fixed):

        with torch.no_grad():
            field_fixed, forward_res = anderson_solver(
                lambda field,interation: self.PIRATE(field, moving, fixed, interation, "forward"), field,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            forward_iter = len(forward_res)
            forward_res = forward_res[-1]
        
        if self.training == True:
            field_hat = self.PIRATE(field_fixed, moving, fixed, forward_iter, "backward")
        else:
            field_hat = self.PIRATE(field_fixed, moving, fixed, forward_iter, "forward")

        return field_hat, forward_iter, forward_res