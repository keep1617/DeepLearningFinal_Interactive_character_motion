# test_udp_receiver.py
import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Listening...")

while True:
    data, addr = sock.recvfrom(65535)
    packet = json.loads(data.decode("utf-8"))

    print("frame:", packet["frame"])
    print("num joints:", len(packet["joints"]))
    print("joint0:", packet["joints"][0])