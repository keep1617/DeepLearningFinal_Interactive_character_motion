import socket
import json
import time
import numpy as np


def stream_npy_motion(motion_path, ip="127.0.0.1", port=5005, fps=20):
    
    ## UDP 소켓 만드는 코드. 

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # SOCKET 만들고 motion 코드 받는다. 
    motion = np.load(motion_path)

    scale = 100.0 # meter to cm
    
    t0 = time.perf_counter()

    print("Streaming:", motion_path)
    print("Shape:", motion.shape)

    for i, joints in enumerate(motion):
        packet = {
            "frame" : i,
            "fps": fps,
            "joints": (joints * scale).tolist()
        }

        sock.sendto(json.dumps(packet).encode("utf-8"), (ip,port))

        if i%20 ==0:
            elapsed = time.perf_counter() - t0
            print(f"frame {i}/{len(motion)} elapsed={elapsed:.2f}s")
        time.sleep(1.0 / fps)

    print("total elapsed:", time.perf_counter() - t0)


