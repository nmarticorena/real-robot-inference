from frankx import Robot

robot  =  Robot("172.16.0.2", repeat_on_error= True, user = "franka", password="franka123")
breakpoint()
print(robot.current_joint_positions())