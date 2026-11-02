import torch
import numpy as np
from model import AvoidNet
from airsim_utils import AirSimDrone
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AvoidNet().to(device)
model.load_state_dict(torch.load("avoid_ckpt.pth",map_location=device))
model.eval()

drone = AirSimDrone()

def preprocess(depth_img):
    depth = cv2.resize(depth,(64,64))
    depth = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
    return depth.to(device)

def inference_loop():
    with torch.no_grad():
        while True:
            d = drone.get_depth_image()
            tensor = preprocess(d)
            pred = model(tensor)
            act = torch.argmax(pred,dim=1).item()
            drone.move(act)

if __name__ == "__main__":
    inference_loop()