import airsim
import numpy as np

class AirSimDrone:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
    def get_depth_image(self):
        res = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        img = res[0]
        depth = np.array(img.image_data_float).reshape(img.height,img.width)
        depth = np.clip(depth,0,20)
        return depth
    def move(self, action):
        vx, vy, vz = 0,0,0
        yaw_rate = 0
        if action == 0:
            vx = 1.5
        elif action == 1:
            yaw_rate = -20
        elif action == 2:
            yaw_rate = 20
        self.client.moveByVelocityBodyFrameAsync(vx,vy,vz,yaw_rate=yaw_rate, duration=0.2)