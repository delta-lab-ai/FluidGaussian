#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import torch

from scene import Scene, GaussianModel
from gaussian_renderer import render

# try to reuse your repo utils (same as training)
from utils.image_utils import psnr as psnr_func

try:
    from utils.loss_utils import ssim as ssim_func
except Exception:
    ssim_func = None

from lpipsPyTorch import lpips, lpips_func


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


@torch.no_grad()
def eval_test_and_write_summary(scene: Scene,
                               gaussians: GaussianModel,
                               pipe,
                               background: torch.Tensor,
                               out_dir: str,
                               iteration: int,
                               no_lpips: bool = False):
    cams = scene.getTestCameras().copy()

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    lpips_model = None
    lpips_model = lpips_func("cuda", net_type="vgg")
    # if (not no_lpips) and (lpips_pkg is not None):
    #     lpips_model = lpips_pkg.LPIPS(net="alex").to(background.device)
    #     lpips_model.eval()

    def _to_lpips(x: torch.Tensor) -> torch.Tensor:
        # expects NCHW in [-1,1]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return (x * 2.0 - 1.0).clamp(-1, 1)

    for cam in cams:
        out = render(cam, gaussians, pipe, background)
        rgb = torch.clamp(out["render"], 0.0, 1.0)  # (3,H,W) in most GS repos

        gt = cam.original_image
        if gt is None:
            continue
        gt = torch.clamp(gt.to(rgb.device), 0.0, 1.0)

        # PSNR
        psnr_vals.append(float(psnr_func(rgb, gt).mean().double().cpu()))

        # SSIM (optional)
        if ssim_func is not None:
            try:
                ssim_vals.append(float(ssim_func(rgb.unsqueeze(0), gt.unsqueeze(0)).detach().cpu()))
            except Exception:
                try:
                    ssim_vals.append(float(ssim_func(rgb, gt).detach().cpu()))
                except Exception:
                    pass

        # LPIPS (optional)
        if lpips_model is not None:
            lp = lpips_model(_to_lpips(rgb), _to_lpips(gt))
            lpips_vals.append(float(lp.mean().detach().cpu()))

    # mean
    psnr_mean = float(np.mean(psnr_vals)) if len(psnr_vals) else None
    ssim_mean = float(np.mean(ssim_vals)) if len(ssim_vals) else None
    lpips_mean = float(np.mean(lpips_vals)) if len(lpips_vals) else None

    ensure_dir(out_dir)
    summary = {
        "iteration": int(iteration),
        "psnr": psnr_mean,
        "ssim": ssim_mean,
        "lpips": lpips_mean,
    }
    with open(os.path.join(out_dir, "summary_occluded.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[TEST SUMMARY]")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser("active_test_summary.py -> testing_result/summary_occluded.json")

    # keep exactly the same arg system as training
    from arguments import ModelParams, PipelineParams, OptimizationParams
    # parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Flags for view selections
    parser.add_argument("--method", type=str, default="rand")
    parser.add_argument("--schema", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--I_test", action="store_true", help="Use I test to get the selection base")
    parser.add_argument("--I_acq_reg", action="store_true", help="apply reg_lambda to acq H too")
    parser.add_argument("--sh_up_every", type=int, default=5_000, help="increase spherical harmonics every N iterations")
    parser.add_argument("--sh_up_after", type=int, default=-1, help="start to increate active_sh_degree after N iterations")
    parser.add_argument("--min_opacity", type=float, default=0.005, help="min_opacity to prune")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"])
    parser.add_argument("--log_every_image", action="store_true", help="log every images during traing")
    # Simulation config
    parser.add_argument("--metric_type", type=str, default="div",
                    help="One of 'div', 'vortex', or 'div,energy' to compute & plot.")
    parser.add_argument("--simulation_scene_config", nargs="+", default=["./configs/sim_config.json", "./configs/sim_config_r01.json"], type=str, 
                        help="Scene config for SPH simulation (fed to run_gs_simulation).")
    parser.add_argument(
        "--sim_directions",
        nargs=3,
        type=float,
        action="append",
        metavar=("DX","DY","DZ"),
        default=[(0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
        help="One or more directions; pass as triplets: --sim_directions 0 1 0  --sim_directions 1 0 0"
    )
    parser.add_argument(
        "--sim_overrides_json",
        type=str,
        default=None,
        help="JSON dict to override keys in simulation scene config, e.g. '{\"velocity_ratio\":6.0, \"timeStepSize\":0.001}'"
    )

    args = parser.parse_args()

    dataset = lp.extract(args)
    dataset.source_path = args.source_path
    dataset.model_path = args.model_path
    dataset.train_json = args.train_json
    dataset.test_json = args.test_json
    dataset.white_background = bool(args.white_background)

    pipe = pp.extract(args)
    pipe.optim = op.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Scene will load gaussians from <model_path>/point_cloud/iteration_xxx/...
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False, load_iteration=args.iterations)
    no_lpips = False
    out_dir = os.path.join(args.model_path, "testing_result")
    eval_test_and_write_summary(
        scene=scene,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        out_dir=out_dir,
        iteration=args.iterations,
        no_lpips=bool(no_lpips),
    )


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
