"""Preview KeeMap retargeted MoMask BVH on a Mixamo FBX in Blender UI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_keemap_retarget_render import (
    clear_scene,
    color_character,
    import_bvh,
    import_fbx,
    parse_color,
    register_keemap,
    run_keemap,
    source_frame_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-dir", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--character-color", default="")
    parser.add_argument("--tts-audio", default="")
    parser.add_argument(
        "--viewport-shading",
        default="MATERIAL",
        choices=("WIREFRAME", "SOLID", "MATERIAL", "RENDERED"),
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.82
    return mat


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=(0, 0, rotation_z))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.color = material.diffuse_color
    obj.data.materials.append(material)
    return obj


def add_sandbag(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    rotation_z: float,
) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=location, rotation=(0, 0, rotation_z))
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.color = material.diffuse_color
    obj.data.materials.append(material)


def create_trench_environment() -> None:
    dirt = make_material("Damp_Dirt", (0.19, 0.15, 0.11, 1.0))
    cut_dirt = make_material("Trench_Cut_Dirt", (0.12, 0.095, 0.07, 1.0))
    sand = make_material("Faded_Sandbags", (0.48, 0.43, 0.32, 1.0))
    wood = make_material("Dark_Wood_Planks", (0.22, 0.15, 0.09, 1.0))
    metal = make_material("Dull_Metal", (0.18, 0.19, 0.18, 1.0))

    add_box("Trench_Floor", (0, 0, -0.04), (2.4, 5.6, 0.08), cut_dirt)
    add_box("Left_Earth_Bank", (-1.7, 0, 0.45), (1.0, 5.8, 0.95), dirt)
    add_box("Right_Earth_Bank", (1.7, 0, 0.45), (1.0, 5.8, 0.95), dirt)
    add_box("Back_Berm", (0, 2.9, 0.22), (4.2, 0.45, 0.5), dirt)

    for y in (-2.2, -1.35, -0.5, 0.35, 1.2, 2.05):
        add_box("Left_Wall_Plank", (-1.08, y, 0.38), (0.08, 0.72, 0.62), wood, rotation_z=0.03)
        add_box("Right_Wall_Plank", (1.08, y, 0.38), (0.08, 0.72, 0.62), wood, rotation_z=-0.03)

    for x in (-0.65, 0.0, 0.65):
        add_box("Duckboard", (x, -1.3, 0.025), (0.18, 1.05, 0.055), wood, rotation_z=0.02)
        add_box("Duckboard", (x, 0.3, 0.025), (0.18, 1.05, 0.055), wood, rotation_z=-0.02)

    for side_x, angle in [(-1.02, 0.08), (1.02, -0.08)]:
        for row, z in enumerate((0.78, 0.98)):
            for i, y in enumerate((-2.1, -1.5, -0.9, 0.9, 1.5, 2.1)):
                add_sandbag(
                    "Sandbag",
                    (side_x, y + (0.16 if row else 0), z),
                    (0.28, 0.16, 0.08),
                    sand,
                    angle + (0.18 if i % 2 else -0.12),
                )

    add_box("Barbed_Wire_Post_Left", (-2.15, -1.7, 1.05), (0.05, 0.05, 0.7), metal, rotation_z=0.2)
    add_box("Barbed_Wire_Post_Right", (2.15, -1.7, 1.05), (0.05, 0.05, 0.7), metal, rotation_z=-0.2)
    add_box("Barbed_Wire_Line", (0, -1.7, 1.25), (4.3, 0.025, 0.025), metal)


def setup_preview(
    source: bpy.types.Object,
    target: bpy.types.Object,
    fps: int,
    viewport_shading: str,
) -> None:
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_set(scene.frame_start)
    scene.world.color = (0.045, 0.052, 0.06)

    source.hide_render = True
    source.hide_viewport = True
    source.hide_set(True)
    target.show_in_front = False

    create_trench_environment()

    bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 4.5))
    light = bpy.context.object
    light.data.energy = 420
    light.data.size = 4

    bpy.ops.object.light_add(type="POINT", location=(-1.2, -1.6, 1.2))
    lantern = bpy.context.object
    lantern.name = "Trench_Lantern"
    lantern.data.energy = 85
    lantern.data.color = (1.0, 0.72, 0.42)

    bpy.ops.object.camera_add(location=(2.7, -4.3, 1.45))
    camera = bpy.context.object
    look_at(camera, Vector((0, 0, 0.78)))
    scene.camera = camera

    for area in bpy.context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        space.shading.type = viewport_shading
        if hasattr(space.shading, "color_type"):
            space.shading.color_type = "MATERIAL"
        if hasattr(space.shading, "use_scene_lights_render"):
            space.shading.use_scene_lights_render = True
        if hasattr(space.shading, "use_scene_world_render"):
            space.shading.use_scene_world_render = True
        if space.region_3d is not None:
            space.region_3d.view_perspective = "CAMERA"

    for obj in bpy.context.scene.objects:
        obj.select_set(False)
        if obj.type == "ARMATURE":
            obj.show_in_front = False
            obj.hide_set(True)

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.find_armature() == target:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            break


def play_audio(audio_path: str) -> None:
    if not audio_path:
        return
    path = Path(audio_path)
    if not path.exists():
        print(f"TTS audio not found: {path}")
        return
    subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def play_audio_on_loop_start(audio_path: str, start_frame: int) -> None:
    if not audio_path:
        return

    state = {"last_frame": None, "played_initial": False}

    def watch_frame() -> float:
        current_frame = bpy.context.scene.frame_current
        last_frame = state["last_frame"]

        if not state["played_initial"]:
            play_audio(audio_path)
            state["played_initial"] = True
        elif last_frame is not None and last_frame > start_frame and current_frame <= start_frame:
            play_audio(audio_path)

        state["last_frame"] = current_frame
        return 0.02

    bpy.app.timers.register(watch_frame, first_interval=0.05)


def main() -> None:
    args = parse_args()
    clear_scene()
    register_keemap(Path(args.addon_dir).resolve())

    source = import_bvh(Path(args.bvh).resolve())
    target = import_fbx(Path(args.fbx).resolve())
    frames = source_frame_count(source)
    if args.max_frames > 0:
        frames = min(frames, args.max_frames)

    run_keemap(source, target, Path(args.mapping).resolve(), frames)
    if args.character_color:
        color_character(target, parse_color(args.character_color))
    setup_preview(source, target, args.fps, args.viewport_shading)

    print(f"KeeMap preview source: {source.name}")
    print(f"KeeMap preview target: {target.name}")
    print(f"KeeMap preview frames: {frames}")
    bpy.context.scene.frame_set(0)
    bpy.ops.screen.animation_play()
    play_audio_on_loop_start(args.tts_audio, bpy.context.scene.frame_start)


if __name__ == "__main__":
    main()
