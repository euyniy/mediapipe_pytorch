from utils import FacialLMBasicBlock, pad_image
from model import FacialFeatures_Model
import numpy as np
import cv2
import torch
import sys


if __name__ == '__main__':
    # load model
    feature_extractor = FacialFeatures_Model()
    weights = torch.load('model_weights/backbone_weights.pth')
    feature_extractor.load_state_dict(weights)
    feature_extractor.eval()

    # resize image TODO: check the centering
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    image = pad_image(image, desired_size=192)

    # extract features
    feats = feature_extractor.predict(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB))
    print(feats)