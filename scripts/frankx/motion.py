from frankx import Affine, LinearRelativeMotion, Robot

robot = Robot("172.16.0.2", repeat_on_error=True, user="franka", password="franka123")
robot.recover_from_errors()
robot.set_dynamic_rel(0.05)

motion = LinearRelativeMotion(Affine(0.2, 0.0, 0.0))
robot.move(motion)
