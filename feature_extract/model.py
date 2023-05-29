import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FacialLMBasicBlock


class FacialFeatures_Model(nn.Module):
    """[MediaPipe facial_landmark model backbone in Pytorch]

    Args:
        nn ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 1x1x1x1404, 1x1x1x1
    # 1x1404x1x1

    def __init__(self):
        """[summary]"""
        super(FacialFeatures_Model, self).__init__()

        self.backbone = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(16),

            FacialLMBasicBlock(16, 16), 
            FacialLMBasicBlock(16, 16),
            FacialLMBasicBlock(16, 32, stride=2), # pad

            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 64, stride=2), 

            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 128, stride=2),

            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128, stride=2),

            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128)
        )


    @torch.no_grad()
    def forward(self, x):        
        x = nn.ReflectionPad2d((1, 0, 1, 0))(x)
        features = self.backbone(x)            
        return features


    def predict(self, img):
        """ single image inference

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))
        
        return self.batch_predict(img.unsqueeze(0))


    def batch_predict(self, x):
        """ batch inference
        currently only single image inference is supported

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        features = self.forward(x)
        return features
   