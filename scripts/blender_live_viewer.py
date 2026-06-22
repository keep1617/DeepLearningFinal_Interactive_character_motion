"""Live Blender viewer for generated MoMask BVH files.

Run this inside Blender UI. It imports the target FBX once, then watches a text
file for new BVH paths. Whenever the path changes it imports the BVH, transfers
animation with KeeMap, and starts timeline playback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


LAST_REQUEST = ""
STATUS_FILE: Path | None = None
SOURCE_OBJECTS: list[str] = []
CURRENT_FRAME = 0
END_FRAME = 0
PLAYING = False
FRAME_INTERVAL = 1 / 20
PLAYBACK_TIMER_REGISTERED = False


def log_status(message: str) -> None:
    print(message)
    if STATUS_FILE is None:
        return
    with STATUS_FILE.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-dir", required=True)
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--fps", type=int, default=20)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def register_keemap(addon_dir: Path) -> None:
    sys.path.insert(0, str(addon_dir.parent.resolve()))
    import KeeMapAnimRetarget

    KeeMapAnimRetarget.register()


def import_fbx(path: Path) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No FBX armature imported from {path}")
    armatures[0].name = "Mixamo_Target"
    return armatures[0]


def import_bvh(path: Path) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(filepath=str(path), frame_start=0, global_scale=1.0)
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No BVH armature imported from {path}")
    armatures[0].name = "MoMask_Source"
    armatures[0].location.x = -1.4
    armatures[0].hide_viewport = False
    armatures[0].hide_render = True
    for obj in imported:
        SOURCE_OBJECTS.append(obj.name)
    return armatures[0]


def delete_previous_source() -> None:
    for name in list(SOURCE_OBJECTS):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
    SOURCE_OBJECTS.clear()


def source_frame_count(source: bpy.types.Object) -> int:
    action = source.animation_data.action if source.animation_data else None
    if action is None:
        raise RuntimeError("BVH source has no animation action")
    start, end = [int(round(v)) for v in action.frame_range]
    return max(1, end - start + 1)


def action_keyframe_count(action: bpy.types.Action | None) -> tuple[int, int]:
    if action is None:
        return 0, 0

    try:
        fcurves = list(action.fcurves)
    except Exception as exc:
        log_status(f"Live viewer: could not inspect action fcurves: {exc}")
        return 0, 0

    return len(fcurves), sum(len(fcurve.keyframe_points) for fcurve in fcurves)


def reset_target(target: bpy.types.Object) -> None:
    target.animation_data_clear()
    for pose_bone in target.pose.bones:
        pose_bone.location = (0, 0, 0)
        pose_bone.rotation_euler = (0, 0, 0)
        pose_bone.rotation_quaternion = (1, 0, 0, 0)
        pose_bone.scale = (1, 1, 1)


def run_keemap(source: bpy.types.Object, target: bpy.types.Object, mapping: Path, frames: int) -> None:
    scene = bpy.context.scene
    settings = scene.keemap_settings
    settings.bone_mapping_file = str(mapping.resolve())
    bpy.ops.wm.keemap_read_file()

    settings.source_rig_name = source.name
    settings.destination_rig_name = target.name
    settings.start_frame_to_apply = 0
    settings.number_of_frames_to_apply = frames
    settings.keyframe_every_n_frames = 1
    settings.bone_rotation_mode = "EULER"

    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    source.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target

    result = bpy.ops.wm.perform_animation_transfer()
    scene.frame_start = 0
    scene.frame_end = frames - 1
    scene.frame_set(0)
    action = target.animation_data.action if target.animation_data else None
    fcurves, keyframes = action_keyframe_count(action)
    log_status(f"Live viewer: KeeMap result = {result}")
    log_status(
        f"Live viewer: target action = {action.name if action else 'NONE'}, "
        f"fcurves = {fcurves}, keyframes = {keyframes}"
    )


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_view(target: bpy.types.Object, fps: int) -> None:
    scene = bpy.context.scene
    scene.render.fps = fps
    target.show_in_front = True

    bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 4.5))
    light = bpy.context.object
    light.data.energy = 550
    light.data.size = 5

    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    plane = bpy.context.object
    mat = bpy.data.materials.new("Ground_Matte")
    mat.diffuse_color = (0.18, 0.18, 0.18, 1)
    plane.data.materials.append(mat)

    bpy.ops.object.camera_add(location=(3.5, -5.0, 1.9))
    camera = bpy.context.object
    look_at(camera, Vector((-0.5, 0, 0.9)))
    scene.camera = camera

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.shading.type = "MATERIAL"
            region_3d = space.region_3d
            region_3d.view_perspective = "CAMERA"

    target.select_set(True)
    bpy.context.view_layer.objects.active = target


def playback_tick() -> float:
    global CURRENT_FRAME, PLAYING
    if not PLAYING:
        return None

    scene = bpy.context.scene
    scene.frame_set(CURRENT_FRAME)
    CURRENT_FRAME += 1
    if CURRENT_FRAME > END_FRAME:
        CURRENT_FRAME = scene.frame_start
    return FRAME_INTERVAL


def start_playback(frames: int, fps: int) -> None:
    global CURRENT_FRAME, END_FRAME, PLAYING, FRAME_INTERVAL, PLAYBACK_TIMER_REGISTERED
    CURRENT_FRAME = 0
    END_FRAME = max(0, frames - 1)
    FRAME_INTERVAL = 1 / fps
    PLAYING = True
    if not PLAYBACK_TIMER_REGISTERED:
        bpy.app.timers.register(playback_tick, first_interval=0.0)
        PLAYBACK_TIMER_REGISTERED = True


def load_motion(bvh_path: Path, target: bpy.types.Object, mapping: Path, fps: int) -> None:
    if not bvh_path.exists():
        log_status(f"Live viewer: BVH does not exist yet: {bvh_path}")
        return

    log_status(f"Live viewer: loading {bvh_path}")
    delete_previous_source()
    reset_target(target)
    source = import_bvh(bvh_path)
    frames = source_frame_count(source)
    action = source.animation_data.action if source.animation_data else None
    _source_fcurves, source_keys = action_keyframe_count(action)
    log_status(f"Live viewer: source frames = {frames}, source keyframes = {source_keys}")
    run_keemap(source, target, mapping, frames)
    start_playback(frames, fps)
    log_status(f"Live viewer: playing {frames} frames with timer playback")


def main() -> None:
    global STATUS_FILE
    args = parse_args()
    request_file = Path(args.request_file)
    mapping = Path(args.mapping)
    STATUS_FILE = request_file.parent / "status.txt"
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text("", encoding="utf-8")

    clear_scene()
    register_keemap(Path(args.addon_dir))
    target = import_fbx(Path(args.fbx))
    setup_view(target, args.fps)

    request_file.parent.mkdir(parents=True, exist_ok=True)
    request_file.write_text("", encoding="utf-8")
    log_status(f"Live viewer ready. Watching {request_file}")

    def watch_request() -> float:
        global LAST_REQUEST
        try:
            request = request_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            log_status(f"Live viewer: cannot read request file: {exc}")
            return 1.0

        if request and request != LAST_REQUEST:
            LAST_REQUEST = request
            log_status(f"Live viewer: request = {request}")
            if "|" in request:
                _nonce, bvh_request = request.split("|", 1)
            else:
                bvh_request = request
            load_motion(Path(bvh_request), target, mapping, args.fps)
        return 0.5

    bpy.app.timers.register(watch_request, first_interval=0.5)


if __name__ == "__main__":
    main()
