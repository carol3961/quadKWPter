import numpy as np
from PyFlyt.core import Aviary

env = Aviary(
    start_pos=np.array([[0.0, 0.0, 1.0]]),
    start_orn=np.array([[0.0, 0.0, 0.0]]),
    drone_type="quadx",
    render=True,
    drone_options={
        "use_camera": True,
        "camera_angle_degrees": 20,
        "camera_FOV_degrees": 90,
        "camera_resolution": (128, 128),
    },
)

# --- simulation loop ---
for step in range(300):
    env.step()

    # capture camera output
    rgba, depth, seg = env.drones[0].camera.capture_image()

    if step % 30 == 0:   # don't spam every step
        print("RGBA:", rgba.shape)
        print("Depth:", depth.shape)
        print("Seg:", seg.shape)

env.close()
