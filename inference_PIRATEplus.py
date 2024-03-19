import torch
import SimpleITK as sitk
from model.base import *
from model.loss import *
from model.PIRATE import *
from model.PIRATEplus import *

def load_checkpoint(model, checkpoint_PATH, device):
    checkpoint_PATH = checkpoint_PATH
    model_CKPT = torch.load(checkpoint_PATH, map_location=device)
    model.load_state_dict(model_CKPT['state_dict'], strict=False)
    print('loading checkpoint!')
    return model

if __name__ == '__main__':
    image_path = "./data"
    model_path = "./pretrained_model/PIRATEplus/OASIS.pth.tar"
    target_path = "./output/"
    
    config_PIRATE = {
    "gamma_inti":5e5,
    "tau_inti":1e-7,
    "iteration":500,
    "image_shape":[160, 192, 224],
    "weight_grad":5e-1
    }
    
    config_PIRATEplus = {
    "max_iter":500,
    "tol":1e-3
    }
    
    image_list = [["moving","fixed"]]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resize = ResizeTransform(1/2, 3)
    resize = resize.to(device)

    denoiser = DnCNN().to(device)
    ForwardIteration = PIRATE(denoiser,config_PIRATE, device)
    PIRATEplus_model = PIRATEplus(ForwardIteration, config_PIRATEplus).to(device)
    PIRATEplus_model = load_checkpoint(PIRATEplus_model, model_path, device)
    
    PIRATEplus_model.eval()
    with torch.no_grad():        
        moving = sitk.ReadImage('./data/moving.nii.gz')
        moving = sitk.GetArrayFromImage(moving)
        fixed = sitk.ReadImage('./data/fixed.nii.gz')
        fixed = sitk.GetArrayFromImage(fixed)
    
        moving = torch.from_numpy(moving).view(1, 1, moving.shape[-3], moving.shape[-2], moving.shape[-1]).to(device)
        fixed = torch.from_numpy(fixed).view(1, 1, fixed.shape[-3], fixed.shape[-2], fixed.shape[-1]).to(device)
        
        field = torch.zeros((1, 3, config_PIRATE['image_shape'][-3]//2,config_PIRATE['image_shape'][-2]//2, config_PIRATE['image_shape'][-1]//2), requires_grad=True, device = device)
        
        field_hat, forward_iter, forward_res = PIRATEplus_model(field, moving, fixed)  
        
        field_full = resize(field_hat)
        
        transformer = SpatialTransformer(config_PIRATE['image_shape'])
        transformer = transformer.to(device)
        warped_image = transformer(moving, field_full, return_phi=False)
            
        warped_np = warped_image.view(warped_image.shape[-3],warped_image.shape[-2],warped_image.shape[-1]).detach().to("cpu")   
        
        out = sitk.GetImageFromArray(warped_np)
        sitk.WriteImage(out,target_path + 'warped_image.nii.gz')