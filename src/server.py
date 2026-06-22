from run_momask import generate_motion
from stream_motion import stream_npy_motion


motion_path = generate_motion(prompt="./prompt.txt", ext="ue_test")
stream_npy_motion(motion_path, ip="127.0.0.1", port=5005, fps=20)