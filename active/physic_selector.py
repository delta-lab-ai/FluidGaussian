import os
import json
import numpy as np
from copy import deepcopy
from typing import List, Dict, Optional, Tuple

import torch
from gaussian_renderer import render
from gs_simulation import run_gs_simulation


class SimSelector(torch.nn.Module):
    """
    Next-Best-View selector driven by simulation-derived per-Gaussian metrics.

    - metric_type: 'div' or 'energy'
    - simulation_scene_config: scene config passed to run_gs_simulation
    - dirs5: 5 fluid directions, default (y+, x+, x-, z+, z-)
    - cache_dir: if provided, read/write cache to avoid repeated simulations
    - use_sigmoid: normalization strategy aligned with compute_view_metric_value
    """
    def __init__(self, args) -> None:
        super().__init__()
        
        self.args = args
        self.seed = args.seed
        self.simulation_scene_config = args.simulation_scene_config
        metric_type = args.metric_type.strip().lower()
        self.metric_type = metric_type
        self.dirs5 = args.sim_directions # [(1, 0, 0)] #(0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)
        self.cache_dir = os.path.join(args.model_path, "sim_cache")
        self.use_sigmoid = True

        # Key names in simulation outputs for different metrics
        if self.metric_type == "div":
            self._metric_key = "mean_divergence"
        elif self.metric_type == "vortex":
            self._metric_key = "mean_vorticity"
        elif self.metric_type == "all":
            self._metric_key = ["mean_divergence", "mean_vorticity"]
        else:
            raise ValueError(f"Unsupported metric_type: {self.metric_type}")

    # ===== Public interface, aligned with RandSelector =====
    @torch.no_grad()
    def nbvs(self, gaussian, scene, num_views: int, pipe=None, background=None, candidate_views=None, **kwargs) -> List[int]:
        """
        Return indices of the top-K training views (from scene.get_candidate_set()),
        ranked by the highest view score.
        Requires pipe/background to render (same as compute_view_metric_value).
        """
        if pipe is None or background is None:
            raise ValueError("SimSelector.nbvs requires pipe and background to render.")

        # 1) Compute (or load) per-Gaussian simulation metrics (averaged over 5 directions)
        # num_candidate_views = list(scene.getTrainCameras())
        # if len(num_candidate_views) >= 5:
        #     config = self.simulation_scene_config[1]
        # else:

        config = self.simulation_scene_config[0]
        if self.metric_type == "all":
            per_g_metric, sim_out, per_g_metric_div, per_g_metric_vortex = self._get_or_compute_per_gaussian_metric(config, gaussian)  # np.ndarray (N,)
        else:   
            per_g_metric, sim_out = self._get_or_compute_per_gaussian_metric(config, gaussian)  # np.ndarray (N,)
        print("max counted particles per Gaussian:", sim_out.get("count_div").max(), (sim_out.get("count_div")>0).sum())
        # 2) For each candidate view, compute the alpha-weighted view score
        if candidate_views is None:
            candidate_views = list(scene.get_candidate_set())
        train_views = list(scene.getTrainCameras())
        # if len(candidate_views) == 3:

        ##### modified for correlation analysis #####
        scores_div: Dict[int, float] = {}
        scores_vortex: Dict[int, float] = {}

        
        if self.metric_type == "all":
            for vid in candidate_views:
                cam = scene.train_cameras[1.0][vid]
                scores_div[vid] = self._compute_view_scalar(cam, gaussian, pipe, background, per_g_metric_div)
            for vid in candidate_views:
                cam = scene.train_cameras[1.0][vid]
                scores_vortex[vid] = self._compute_view_scalar(cam, gaussian, pipe, background, per_g_metric_vortex)
            dataset = self.args.source_path.split("/")[-1]
            out_txt = os.path.join(self.cache_dir, f"view_div_vor_{len(train_views)}_{dataset}.txt")
            vids = sorted(set(scores_div.keys()) & set(scores_vortex.keys()))
            div_vals = np.array([scores_div[v] for v in vids], dtype=np.float64)
            vor_vals = np.array([scores_vortex[v] for v in vids], dtype=np.float64)
            if len(vids) < 2 or np.std(div_vals) == 0 or np.std(vor_vals) == 0:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(div_vals, vor_vals)[0, 1])
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(f"# pearson_corr(div,vortex)\t{corr:.8f}\n")
                f.write("vid\tdiv\tvortex\n")
                for v in vids:
                    f.write(f"{v}\t{scores_div[v]:.10g}\t{scores_vortex[v]:.10g}\n")
            print(f"[OK] wrote: {out_txt}  (pearson_corr={corr:.6f})")
        # import pdb; pdb.set_trace()
        
        scores: Dict[int, float] = {}
        for vid in candidate_views:
            cam = scene.train_cameras[1.0][vid]
            scores[vid] = self._compute_view_scalar(cam, gaussian, pipe, background, per_g_metric)
        ########### added #########################################
        top = sorted(candidate_views, key=lambda i: scores[i], reverse=True)
        return top[:num_views]
        #########################################################


        # # 3) Select top-K
        # top = sorted(candidate_views, key=lambda i: scores[i], reverse=True)
        # print(top)
        # # import pdb; pdb.set_trace()
        # return top[:num_views]

    # ===== Internal: per-Gaussian metric from simulation =====
    def _get_or_compute_per_gaussian_metric(self, config, gaussians, device="cuda") -> np.ndarray:
        """
        Run run_gs_simulation over 5 directions (or load from cache),
        extract self._metric_key, align lengths, and average.
        Return np.float32 array of shape (N,).
        """
        os.makedirs(self.cache_dir, exist_ok=True) if self.cache_dir else None
        mean_list = []
        mean_list_div = []
        mean_list_vortex = []

        for i, fd in enumerate(self.dirs5):
            # 1) Try cache
            cached_pt = None
            if self.cache_dir:
                cached_pt = os.path.join(self.cache_dir, f"uncert_{i}_{self.metric_type}_sim_out.pt") 
                # if os.path.isfile(cached_pt):
                #     try:
                #         sim_out = torch.load(cached_pt, map_location="cpu")
                #     except Exception:
                #         sim_out = None
                # else:
            #     sim_out = None
            # else:
            sim_out = None

            # 2) Run simulation if no cache is available
            if sim_out is None:
                sim_out = run_gs_simulation(
                    config,
                    gaussians,
                    fluid_dir=fd,
                    device="gpu" if device.startswith("cuda") else "cpu",
                    out_dir=(self.cache_dir and os.path.join(self.cache_dir, f"uncert_{i}_{self.metric_type}")) or None,
                    args=self.args,
                )
                if self.cache_dir:
                    torch.save(sim_out, cached_pt)

            # 3) Extract metric array
            if self.metric_type == "all":
                arr_div = np.asarray(sim_out["mean_divergence"], dtype=np.float32)
                arr_vortex = np.asarray(sim_out["mean_vorticity"], dtype=np.float32)
                arr = 0.5 * (arr_div + arr_vortex)
                mean_list_div.append(arr_div)
                mean_list_vortex.append(arr_vortex)
                
            else:
                if self._metric_key not in sim_out:
                    raise KeyError(f"Simulation output missing key '{self._metric_key}'. Available: {list(sim_out.keys())}")
                arr = np.asarray(sim_out[self._metric_key], dtype=np.float32)
            mean_list.append(arr)

        # 4) Align lengths and average
        minN = min(m.shape[0] for m in mean_list)
        per_g_metric = np.max([m[:minN] for m in mean_list], axis=0).astype(np.float32)
        if self.metric_type == "all":
            per_g_metric_div = np.max([m[:minN] for m in mean_list_div], axis=0).astype(np.float32)
            per_g_metric_vortex = np.max([m[:minN] for m in mean_list_vortex], axis=0).astype(np.float32)
            return per_g_metric, sim_out, per_g_metric_div, per_g_metric_vortex

        # per_g_metric = np.sum([m[:minN] for m in mean_list], axis=0).astype(np.float32)
        print("gs div mean", per_g_metric.mean())
        return per_g_metric, sim_out


    @torch.no_grad()
    def _compute_view_scalar(
        self,
        cam,
        gaussians,
        pipe,
        background: torch.Tensor,
        per_gaussian_metric: np.ndarray,
        eps: float = 1e-8,
        mode: str = "soft", 
    ) -> float:
       
        device = background.device
        x_np = np.abs(np.asarray(per_gaussian_metric, dtype=np.float32))
        N_model = int(gaussians._features_dc.shape[0])

        if x_np.shape[0] != N_model:
            n_copy = min(x_np.shape[0], N_model)
            aligned = np.zeros((N_model,), dtype=np.float32)
            aligned[:n_copy] = x_np[:n_copy]
            x_np = aligned  # pad/truncate

        x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)  # (N,)

        out = render(cam, gaussians, pipe, background)

        vis_mask = out.get("visibility_filter", None)
        if vis_mask is None:
            radii = out.get("radii", None)
            if radii is None:
                vis_mask = torch.ones_like(x_t, dtype=torch.bool)
                radii = None
            else:
                if isinstance(radii, np.ndarray):
                    radii = torch.from_numpy(radii)
                radii = radii.to(device=device)
                vis_mask = radii > 0
        else:
            if isinstance(vis_mask, np.ndarray):
                vis_mask = torch.from_numpy(vis_mask)
            vis_mask = vis_mask.to(device=device, dtype=torch.bool)
            radii = out.get("radii", None)
            if isinstance(radii, np.ndarray):
                radii = torch.from_numpy(radii).to(device=device)
            elif radii is not None:
                radii = radii.to(device=device)

        metric_img = out.get("render", None)
        if metric_img is not None:
            H = metric_img.shape[-2]
            W = metric_img.shape[-1]
        else:
            H, W = 1024, 1024

        x_vis = x_t[vis_mask]

        if mode == "soft" and (radii is not None):
            r_vis = radii[vis_mask].float()
            area = 3.141592653589793 * (r_vis ** 2)            
            w = (area / float(H * W + eps)).clamp_min(0.0).clamp_max(1.0)
            score = (x_vis * w).sum()
            # score = (x_vis * w).mean()
        else:
            score = x_vis.sum()
            # score = x_vis.mean()

        return float(score.double().cpu().item())