from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .V_sel import VarSelector
# from .combine_selector import CombSelector
from .physic_selector import SimSelector
import torch
from scene import Scene
from typing import List
import random

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "variance": VarSelector, "physics": SimSelector}

class CombActiveSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        
        self.seed = args.seed
        self.args = args
        # self.hreg_selector = HRegSelector(args)
        # self.vsel_selector = VarSelector(args)

    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        # hreg_idxs = self.hreg_selector.nbvs(gaussians, scene, num_views, pipe, background, exit_func)
        # vsel_idxs = self.vsel_selector.nbvs(gaussians, scene, num_views, pipe, background, exit_func)
        # candidate_cameras = scene.getCandidateCameras()
        candidate_views = list(scene.getTrainCameras())

        # num_views_new = max(2, 20-len(candidate_views)*2)
        # num_views_new = max(2, 20-len(candidate_views)**2)
        num_views_new = max(2, 10-len(candidate_views))
        # num_views_new = 3


        print("num_views of fisherrf:", num_views_new)
        # if len(candidate_views) <= 5:
        #     return physic_selector.nbvs(gaussians, scene, num_views, pipe, background, candidate_views=candidate_views, exit_func=exit_func)
        # selectors = [methods_dict[m](self.args) for m in ["variance"]]
        selectors = [methods_dict[m](self.args) for m in ["variance"]]
        idxs = []
        for selector in selectors:
            idxs.extend(selector.nbvs(gaussians, scene, num_views_new, pipe, background, exit_func))
        all_top_k_idxs = list(set(idxs))
        physic_selector = methods_dict["physics"](self.args)

        physic_idxs = physic_selector.nbvs(gaussians, scene, num_views, pipe, background, candidate_views=all_top_k_idxs, exit_func=exit_func)
        # currently random, adding metric here further.
        # available_idxs = set(hreg_idxs).union(vsel_idxs)
        # selected_idxs  = random.sample(list(available_idxs), num_views)
        return physic_idxs