"""Realtime UDP joint viewer for MoMask npy streams.

This Blender script listens for JSON packets from ``src/stream_motion.py`` and
updates a simple 22-joint skeleton every received frame. It avoids BVH import,
KeeMap transfer, rendering, and ffmpeg so motion appears as soon as frames are
streamed.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import bpy
from mathutils import Vector


T2M_CHAINS = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
EDGES = sorted({tuple(sorted((a, b))) for chain in T2M_CHAINS for a, b in zip(chain, chain[1:])})

SOCK: socket.socket | None = None
SKELETON: bpy.types.Object | None = None
JOINT_OBJECTS: list[bpy.types.Object] = []
STATUS_FILE: Path | None = None
SCALE = 0.01
FRAME_COUNT = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--status-file", default="")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(message)
    if STATUS_FILE is not None:
        with STATUS_FILE.open("a", encoding="utf-8") as file:
            file.write(message + "\n")


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_scene() -> None:
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = 200

    bpy.ops.object.light_add(type="AREA", location=(0, -4, 5))
    light = bpy.context.object
    light.data.energy = 450
    light.data.size = 5

    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    plane = bpy.context.object
    mat = bpy.data.materials.new("Ground_Matte")
    mat.diffuse_color = (0.18, 0.18, 0.18, 1)
    plane.data.materials.append(mat)

    bpy.ops.object.camera_add(location=(2.6, -4.5, 1.8), rotation=(1.25, 0, 0.52))
    scene.camera = bpy.context.object


def create_joint_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("Live_Joints")
    mat.diffuse_color = (0.1, 0.75, 1.0, 1.0)
    return mat


def create_bone_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("Live_Bones")
    mat.diffuse_color = (0.95, 0.95, 0.95, 1.0)
    return mat


def create_skeleton() -> None:
    global SKELETON, JOINT_OBJECTS

    mesh = bpy.data.meshes.new("LiveSkeletonMesh")
    mesh.from_pydata([(0, 0, 0)] * 22, EDGES, [])
    mesh.update()

    SKELETON = bpy.data.objects.new("Live_MoMask_Skeleton", mesh)
    bpy.context.collection.objects.link(SKELETON)
    SKELETON.show_in_front = True
    SKELETON.data.materials.append(create_bone_material())

    joint_mat = create_joint_material()
    JOINT_OBJECTS = []
    for index in range(22):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.035, location=(0, 0, 0))
        joint = bpy.context.object
        joint.name = f"joint_{index:02d}"
        joint.data.materials.append(joint_mat)
        JOINT_OBJECTS.append(joint)


def convert_joint(raw_joint: list[float]) -> Vector:
    x, y, z = raw_joint
    return Vector((x * SCALE, -z * SCALE, y * SCALE))


def update_skeleton(joints: list[list[float]]) -> None:
    if SKELETON is None:
        return
    if len(joints) < 22:
        log(f"UDP viewer: expected 22 joints, got {len(joints)}")
        return

    positions = [convert_joint(joints[index]) for index in range(22)]
    root = positions[0].copy()
    positions = [position - root for position in positions]

    for vertex, position in zip(SKELETON.data.vertices, positions):
        vertex.co = position
    SKELETON.data.update()

    for obj, position in zip(JOINT_OBJECTS, positions):
        obj.location = position


def udp_tick() -> float:
    global FRAME_COUNT
    if SOCK is None:
        return 0.05

    received = 0
    while received < 20:
        try:
            data, _addr = SOCK.recvfrom(65535)
        except BlockingIOError:
            break

        packet = json.loads(data.decode("utf-8"))
        update_skeleton(packet["joints"])
        FRAME_COUNT = int(packet.get("frame", FRAME_COUNT))
        bpy.context.scene.frame_set(FRAME_COUNT)
        received += 1

    return 0.01


def main() -> None:
    global SOCK, STATUS_FILE, SCALE
    args = parse_args()
    SCALE = args.scale
    if args.status_file:
        STATUS_FILE = Path(args.status_file)
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text("", encoding="utf-8")

    clear_scene()
    setup_scene()
    create_skeleton()

    SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    SOCK.bind((args.host, args.port))
    SOCK.setblocking(False)

    log(f"UDP viewer ready on {args.host}:{args.port}, scale={SCALE}")
    bpy.app.timers.register(udp_tick, first_interval=0.05)


if __name__ == "__main__":
    main()
