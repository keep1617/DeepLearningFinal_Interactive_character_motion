import subprocess
import argparse

from pathlib import Path



MOMASK_DIR = Path("../3rdParty/momask-codes")


def generate_motion(prompt: str, ext:str="ue_stream"):
    cmd = [
        "python", "gen_t2m.py",
        "--gpu_id", "0",
        "--ext", ext,
        "--text_prompt", prompt,
    ]

    subprocess.run(cmd, cwd=MOMASK_DIR, check=True)

    animation_dir = MOMASK_DIR / "generation" / ext / "animations"/ "0"
    bvh_files = sorted(animation_dir.glob("*_ik.bvh"))

    if not bvh_files:
        raise FileNotFoundError("No generated .npy motion found.")
    

    return bvh_files[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_file", type=str)
    parser.add_argument("--ext", type=str, default="exp1")

    args = parser.parse_args()

    prompt_path = Path(args.prompt_file)
    prompt = prompt_path.read_text(encoding="utf-8").strip()


    result = generate_motion(prompt, args.ext)







