"""Use KeeMapRig to retarget MoMask BVH onto a Mixamo FBX and render PNG frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-dir", default="3rdParty/KeeMapAnimRetarget")
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    parser.add_argument(
        "--character-color",
        default="",
        help="Optional material override for character meshes as r,g,b or #RRGGBB.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def register_keemap(addon_dir: Path) -> None:
    sys.path.insert(0, str(addon_dir.parent.resolve()))
    import KeeMapAnimRetarget

    KeeMapAnimRetarget.register()


def import_bvh(path: Path) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(filepath=str(path), frame_start=0, global_scale=1.0)
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No BVH armature imported from {path}")
    armatures[0].name = "MoMask_Source"
    return armatures[0]


def import_fbx(path: Path) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No FBX armature imported from {path}")
    armatures[0].name = "Mixamo_Target"
    return armatures[0]


def source_frame_count(source: bpy.types.Object) -> int:
    action = source.animation_data.action if source.animation_data else None
    if action is None:
        raise RuntimeError("BVH source has no animation action")
    start, end = [int(round(v)) for v in action.frame_range]
    return max(1, end - start + 1)


def parse_color(value: str) -> tuple[float, float, float, float]:
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return (
            int(value[1:3], 16) / 255.0,
            int(value[3:5], 16) / 255.0,
            int(value[5:7], 16) / 255.0,
            1.0,
        )
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--character-color must be r,g,b or #RRGGBB")
    return (parts[0], parts[1], parts[2], 1.0)


def color_character(target: bpy.types.Object, color: tuple[float, float, float, float]) -> None:
    mat = bpy.data.materials.new("Character_Color")
    mat.diffuse_color = color

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.find_armature() != target:
            continue
        obj.color = color
        obj.data.materials.clear()
        obj.data.materials.append(mat)


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

    bpy.ops.wm.perform_animation_transfer()
    scene.frame_start = 0
    scene.frame_end = frames - 1


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_render(source: bpy.types.Object, output: Path, fps: int, width: int, height: int) -> None:
    scene = bpy.context.scene
    output.parent.mkdir(parents=True, exist_ok=True)

    source.hide_render = True
    source.hide_viewport = True

    bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 4.5))
    light = bpy.context.object
    light.data.energy = 550
    light.data.size = 5

    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    plane = bpy.context.object
    mat = bpy.data.materials.new("Ground_Matte")
    mat.diffuse_color = (0.18, 0.18, 0.18, 1)
    plane.data.materials.append(mat)

    bpy.ops.object.camera_add(location=(2.8, -4.2, 1.7))
    camera = bpy.context.object
    look_at(camera, Vector((0, 0, 0.9)))
    scene.camera = camera

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.light = "FLAT"
    scene.render.fps = fps
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filepath = str(output.resolve())
    scene.render.image_settings.file_format = "PNG"


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
    setup_render(source, Path(args.output), args.fps, args.resolution_x, args.resolution_y)

    print(f"KeeMap source: {source.name}")
    print(f"KeeMap target: {target.name}")
    print(f"KeeMap frames: {frames}")
    print(f"Rendering: {args.output}")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
