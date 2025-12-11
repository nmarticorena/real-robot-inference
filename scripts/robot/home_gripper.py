import os
from panda_py.libfranka import Gripper

gripper = Gripper(os.environ["PANDA_IP"])

gripper.homing()
