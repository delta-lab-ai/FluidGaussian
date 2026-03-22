import os, taichi as ti, numpy as np
from SPH.utils import SimConfig
from SPH.containers import DFSPHContainer
from SPH.fluid_solvers import DFSPHSolver

from tqdm import tqdm

def run_gs_simulation(scene_file: str, gaussians, fluid_dir, device: str = "gpu",
                      out_dir=None, args=None, ret_step_div=False):
    ti.init(arch=ti.cuda if device == "gpu" else ti.cpu, device_memory_fraction=0.8)
    cfg = SimConfig(scene_file_path=scene_file)
    fps = cfg.get_cfg("fps") or 60
    dt  = float(cfg.get_cfg("timeStepSize"))
    total_time = cfg.get_cfg("totalTime") or 10.0
    total_rounds = int(total_time / dt)
    output_frames = bool(cfg.get_cfg("exportFrame"))
    frame_time = 1.0 / fps
    output_interval = int((cfg.get_cfg("outputInterval") or (frame_time / dt)))

    if args and getattr(args, "sim_overrides", None):
        print("[Info] Overriding simulation config with provided args.sim_overrides:")
        for k, v in args.sim_overrides.items():
            print("replacing", k, "current value:", cfg.get_cfg(k), "with", v)
            cfg.set_cfg(k, v, enforce_exist=True)

    container = DFSPHContainer(cfg, GGUI=True, gaussians=gaussians, fluid_dir=fluid_dir)
    solver = DFSPHSolver(container)
    solver.prepare()

    if ret_step_div:
        divs_step = []
    dim = 3
    if output_frames:
        window = ti.ui.Window('SPH', (1024, 1024), show_window=False, vsync=False)
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(6.0, 3.0, 2.0); camera.up(0.0, 1.0, 0.0); camera.lookat(0.0, 1.5, 2.0); camera.fov(90)
        scene.set_camera(camera)
        canvas = window.get_canvas()
        look_from = 'z'
        if hasattr(container, "domain_start") and hasattr(container, "domain_end"):
            domain_start = np.array(container.domain_start, dtype=np.float32)
            domain_end   = np.array(container.domain_end,   dtype=np.float32)
            cam_p_y = (domain_start[1] + domain_end[1]) / 2.0 + 1.0
            cam_lk_at_y = (domain_start[1] + domain_end[1]) / 2.0
            if look_from == 'z':
                cam_p_x = 8.5
                cam_lk_at_x = 0.0
                cam_p_z = (domain_start[2] + domain_end[2]) / 2.0
                cam_lk_at_z = (domain_start[2] + domain_end[2]) / 2.0
            elif look_from == 'x':
                cam_p_x = (domain_start[0] + domain_end[0]) / 2.0
                cam_lk_at_x = (domain_start[0] + domain_end[0]) / 2.0
                cam_p_z = 8.5
                cam_lk_at_z = 0.0
            camera.position(cam_p_x, cam_p_y, cam_p_z); camera.up(0.0, 1.0, 0.0); camera.lookat(cam_lk_at_x, cam_lk_at_y, cam_lk_at_z); camera.fov(90)
        else:
            domain_start = np.array(cfg.get_cfg("domainStart"), dtype=np.float32)
            domain_end   = np.array(cfg.get_cfg("domainEnd"),   dtype=np.float32)

        if dim == 3:
            x0, y0, z0 = domain_start.tolist()
            x1, y1, z1 = domain_end.tolist()
            box_anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
            box_anchors[0] = ti.Vector([x0, y0, z0])
            box_anchors[1] = ti.Vector([x0, y1, z0])
            box_anchors[2] = ti.Vector([x1, y0, z0])
            box_anchors[3] = ti.Vector([x1, y1, z0])
            box_anchors[4] = ti.Vector([x0, y0, z1])
            box_anchors[5] = ti.Vector([x0, y1, z1])
            box_anchors[6] = ti.Vector([x1, y0, z1])
            box_anchors[7] = ti.Vector([x1, y1, z1])

        box_lines_indices = ti.field(ti.i32, shape=24)
        for i, v in enumerate([0,1, 0,2, 1,3, 2,3, 4,5, 4,6, 5,7, 6,7, 0,4, 1,5, 2,6, 3,7]):
            box_lines_indices[i] = v

    for cnt in tqdm(range(total_rounds), desc="Simulation", ncols=100):
        solver.step()

        solver.compute_velocity_divergence()
        solver.compute_vorticity()
        solver.compute_uniform_vorticity()
        if container.num_gauss > 0:
            solver.compute_cfl_number(dt=dt)
            solver.accumulate_cfl_around_gaussians(solver.container.num_gauss)
            solver.accumulate_vorticity_around_gaussians(container.num_gauss)
            solver.accumulate_divergence_around_gaussians(container.num_gauss)

        container.copy_to_vis_buffer(invisible_objects=cfg.get_cfg("invisibleObjects") or [], dim=dim)

        if output_frames and (cnt % output_interval == 0):
            if dim == 3:
                scene.set_camera(camera)
                scene.point_light((cam_lk_at_x, cam_lk_at_y, cam_lk_at_z), color=(5.0, 5.0, 5.0))
                scene.ambient_light((0.2, 0.2, 0.2))
                scene.particles(container.x_vis_buffer, radius=container.dx,
                                per_vertex_color=container.color_vis_buffer)
                scene.lines(box_anchors, indices=box_lines_indices,
                            color=(0.99, 0.68, 0.28), width=1.0)
                canvas.scene(scene)
            else:
                canvas.set_background_color((0, 0, 0))
                canvas.circles(container.x_vis_buffer, radius=container.dx / 80.0, color=(1, 1, 1))
            os.makedirs(out_dir, exist_ok=True)
            window.save_image(f"{out_dir}/{cnt:06d}_raw_view.png")

        if ret_step_div:
            sum_div   = container.gauss_sum_div.to_numpy()
            count_div = container.gauss_count.to_numpy()
            mean_div  = np.divide(sum_div, np.maximum(count_div, 1), dtype=np.float64)
            divs_step.append(sum_div.mean())

        # if cnt % 10 == 0:
        #     pos = container.particle_positions.to_numpy()[:container.fluid_particle_num[None]]
        #     y = pos[:, 1]
        #     print(cnt, "y mean:", y.mean(), "min:", y.min(), "max:", y.max())

    sum_div   = container.gauss_sum_div.to_numpy()
    count_div = container.gauss_count.to_numpy()
    mean_div  = np.divide(sum_div, np.maximum(count_div, 1), dtype=np.float64)

    sum_vor   = container.gauss_sum_vor.to_numpy()
    count_vor = container.gauss_count_vor.to_numpy()
    mean_vor  = np.divide(sum_vor, np.maximum(count_vor, 1), dtype=np.float64)

    sum_cfl   = container.gauss_sum_cfl.to_numpy()
    count_cfl = container.gauss_count_cfl.to_numpy()
    mean_cfl  = np.divide(sum_cfl, np.maximum(count_cfl, 1), dtype=np.float64)

    print("divergence means:", mean_div.mean())
    if ret_step_div:
        return {
            "mean_divergence": mean_div,
            "count_div":       count_div,
            "divs_step":       divs_step,
            "mean_vorticity":  mean_vor,
            "count_vor":       count_vor,
            "mean_cfl":        mean_cfl,
            "count_cfl":       count_cfl,
        }
    return {
        "mean_divergence": mean_div,
        "count_div":       count_div,
        "mean_vorticity":  mean_vor,
        "count_vor":       count_vor,
        "mean_cfl":        mean_cfl,
        "count_cfl":       count_cfl,
    }
