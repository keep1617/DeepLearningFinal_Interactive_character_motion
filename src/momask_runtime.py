#!/usr/bin/env python3
"""Reusable MoMask text-to-motion runtime.

This keeps the heavy MoMask models loaded in memory so demo code can call
``generate(prompt)`` repeatedly without spawning ``gen_t2m.py`` each turn.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/momask-matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


ROOT = Path(__file__).resolve().parents[1]
MOMASK_DIR = ROOT / "3rdParty" / "momask-codes"

if str(MOMASK_DIR) not in sys.path:
    sys.path.insert(0, str(MOMASK_DIR))

from gen_t2m import (  # noqa: E402
    load_len_estimator,
    load_res_model,
    load_trans_model,
    load_vq_model,
)
from utils.fixseed import fixseed  # noqa: E402
from utils.get_opt import get_opt  # noqa: E402
from utils.motion_process import recover_from_ric  # noqa: E402
from utils.paramUtil import t2m_kinematic_chain  # noqa: E402
from utils.plot_script import plot_3d_motion  # noqa: E402
from visualization.joints2bvh import Joint2BVHConvertor  # noqa: E402


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass
class MomaskConfig:
    momask_dir: Path = MOMASK_DIR
    checkpoints_dir: Path = MOMASK_DIR / "checkpoints"
    dataset_name: str = "t2m"
    name: str = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
    res_name: str = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"
    gpu_id: int = -1
    ext: str = "demo_run"
    seed: int = 10107
    repeat_times: int = 1
    cond_scale: float = 4.0
    temperature: float = 1.0
    topkr: float = 0.9
    time_steps: int = 18
    gumbel_sample: bool = False
    motion_length: int = 0
    write_preview_mp4: bool = True


class MomaskRuntime:
    def __init__(self, config: MomaskConfig | None = None, **overrides: Any) -> None:
        self.config = config or MomaskConfig()
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown MomaskConfig field: {key}")
            setattr(self.config, key, value)

        fixseed(self.config.seed)
        self.device = torch.device(
            "cpu" if self.config.gpu_id == -1 else f"cuda:{self.config.gpu_id}"
        )
        if self.config.gpu_id != -1:
            torch.cuda.set_device(self.config.gpu_id)

        self.result_dir = self.config.momask_dir / "generation" / self.config.ext
        self.joints_dir = self.result_dir / "joints"
        self.animation_dir = self.result_dir / "animations"
        self.joints_dir.mkdir(parents=True, exist_ok=True)
        self.animation_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = str(self.config.checkpoints_dir)
        root_dir = self.config.checkpoints_dir / self.config.dataset_name / self.config.name
        model_opt_path = root_dir / "opt.txt"
        self.model_opt = get_opt(
            str(model_opt_path),
            device=self.device,
            checkpoints_dir=checkpoints_dir,
        )

        dim_pose = 251 if self.config.dataset_name == "kit" else 263
        vq_opt_path = (
            self.config.checkpoints_dir
            / self.config.dataset_name
            / self.model_opt.vq_name
            / "opt.txt"
        )
        self.vq_opt = get_opt(
            str(vq_opt_path),
            device=self.device,
            checkpoints_dir=checkpoints_dir,
        )
        self.vq_opt.dim_pose = dim_pose
        self.vq_model, self.vq_opt = load_vq_model(self.vq_opt)

        self.model_opt.num_tokens = self.vq_opt.nb_code
        self.model_opt.num_quantizers = self.vq_opt.num_quantizers
        self.model_opt.code_dim = self.vq_opt.code_dim

        res_opt_path = (
            self.config.checkpoints_dir
            / self.config.dataset_name
            / self.config.res_name
            / "opt.txt"
        )
        self.res_opt = get_opt(
            str(res_opt_path),
            device=self.device,
            checkpoints_dir=checkpoints_dir,
        )
        self.res_model = load_res_model(self.res_opt, self.vq_opt, self._runtime_opt())
        self.t2m_transformer = load_trans_model(
            self.model_opt,
            self._runtime_opt(),
            "latest.tar",
        )
        self.length_estimator = load_len_estimator(self.model_opt)

        self.res_model.eval().to(self.device)
        self.t2m_transformer.eval().to(self.device)
        self.vq_model.eval().to(self.device)
        self.length_estimator.eval().to(self.device)

        meta_dir = (
            self.config.checkpoints_dir
            / self.config.dataset_name
            / self.model_opt.vq_name
            / "meta"
        )
        self.mean = np.load(meta_dir / "mean.npy")
        self.std = np.load(meta_dir / "std.npy")
        self.nb_joints = 21 if self.config.dataset_name == "kit" else 22
        with pushd(self.config.momask_dir):
            self.converter = Joint2BVHConvertor()

    def _runtime_opt(self) -> Any:
        class RuntimeOpt:
            pass

        opt = RuntimeOpt()
        opt.name = self.config.name
        opt.device = self.device
        opt.gumbel_sample = self.config.gumbel_sample
        return opt

    def _estimate_token_lengths(self, prompts: list[str]) -> torch.Tensor:
        text_embedding = self.t2m_transformer.encode_text(prompts)
        pred_dis = self.length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)
        return Categorical(probs).sample().long()

    def _inv_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def generate(
        self,
        prompt: str,
        *,
        motion_length: int | None = None,
        ext: str | None = None,
        write_preview_mp4: bool | None = None,
    ) -> dict[str, Path | str | int]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        if ext is not None and ext != self.config.ext:
            self.config.ext = ext
            self.result_dir = self.config.momask_dir / "generation" / self.config.ext
            self.joints_dir = self.result_dir / "joints"
            self.animation_dir = self.result_dir / "animations"
            self.joints_dir.mkdir(parents=True, exist_ok=True)
            self.animation_dir.mkdir(parents=True, exist_ok=True)

        target_motion_length = (
            self.config.motion_length if motion_length is None else motion_length
        )
        should_write_mp4 = (
            self.config.write_preview_mp4
            if write_preview_mp4 is None
            else write_preview_mp4
        )

        prompts = [prompt]
        if target_motion_length == 0:
            token_lens = self._estimate_token_lengths(prompts).to(self.device)
        else:
            token_lens = torch.LongTensor([target_motion_length // 4]).to(self.device)

        m_lengths = token_lens * 4
        sample_index = 0
        repeat_index = 0

        with torch.no_grad():
            mids = self.t2m_transformer.generate(
                prompts,
                token_lens,
                timesteps=self.config.time_steps,
                cond_scale=self.config.cond_scale,
                temperature=self.config.temperature,
                topk_filter_thres=self.config.topkr,
                gsample=self.config.gumbel_sample,
            )
            mids = self.res_model.generate(
                mids,
                prompts,
                token_lens,
                temperature=1,
                cond_scale=5,
            )
            pred_motions = self.vq_model.forward_decoder(mids).detach().cpu().numpy()
            data = self._inv_transform(pred_motions)

        joint_data = data[0][: m_lengths[0]]
        joint = recover_from_ric(
            torch.from_numpy(joint_data).float(),
            self.nb_joints,
        ).numpy()

        animation_path = self.animation_dir / str(sample_index)
        joint_path = self.joints_dir / str(sample_index)
        animation_path.mkdir(parents=True, exist_ok=True)
        joint_path.mkdir(parents=True, exist_ok=True)

        length = int(m_lengths[0].item())
        stem = f"sample{sample_index}_repeat{repeat_index}_len{length}"
        ik_bvh_path = animation_path / f"{stem}_ik.bvh"
        bvh_path = animation_path / f"{stem}.bvh"
        npy_path = joint_path / f"{stem}.npy"

        _, ik_joint = self.converter.convert(
            joint,
            filename=str(ik_bvh_path),
            iterations=100,
        )
        _, joint = self.converter.convert(
            joint,
            filename=str(bvh_path),
            iterations=100,
            foot_ik=False,
        )
        np.save(npy_path, joint)

        ik_mp4_path = animation_path / f"{stem}_ik.mp4"
        mp4_path = animation_path / f"{stem}.mp4"
        if should_write_mp4:
            plot_3d_motion(
                str(ik_mp4_path),
                t2m_kinematic_chain,
                ik_joint,
                title=prompt,
                fps=20,
            )
            plot_3d_motion(
                str(mp4_path),
                t2m_kinematic_chain,
                joint,
                title=prompt,
                fps=20,
            )

        return {
            "prompt": prompt,
            "length": length,
            "npy": npy_path,
            "bvh": bvh_path,
            "ik_bvh": ik_bvh_path,
            "mp4": mp4_path,
            "ik_mp4": ik_mp4_path,
        }


if __name__ == "__main__":
    runtime = MomaskRuntime()
    result = runtime.generate("a person is waving both hands.")
    for key, value in result.items():
        print(f"{key}: {value}")
