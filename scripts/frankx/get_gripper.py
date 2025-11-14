from frankx import Gripper

gripper = Gripper("172.16.0.2")
# gripper.close(blocking = True)
gripper.open(blocking = True)
while True:
    print(gripper.width())

