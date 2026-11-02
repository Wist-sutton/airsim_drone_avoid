import rospy
from inference import inference_loop

if __name__ == "__main__":
    rospy.init_node("drone_avoid_node")
    inference_loop()