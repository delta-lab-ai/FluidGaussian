# modified from https://github.com/erizmr/SPH_Taichi/blob/4a701fd1397a7da30eb7b799017614209f04804d/particle_system.py
import taichi as ti
import numpy as np
import trimesh as tm
from tqdm import tqdm
from functools import reduce
from ..utils import SimConfig
from ..gs_utils import _gs_to_voxel_points
from tqdm import tqdm
import pdb

def _axis_from_dir(fluid_dir):
    fd = np.asarray(fluid_dir, dtype=float)
    if fd.shape != (3,):
        raise ValueError(f"fluid_dir must be (3,), got {fd.shape}")
    if np.allclose(fd, 0):
        raise ValueError("fluid_dir cannot be zero vector.")
    axis = int(np.argmax(np.abs(fd)))
    sign = 1 if fd[axis] >= 0 else -1
    return axis, sign

def _double_span(min_v, max_v):
    c = 0.5 * (min_v + max_v)
    half = 0.5 * (max_v - min_v)
    half2 = 2.0 * half
    return c, c + half2*2


@ti.func
def _clampi(v: ti.i32, lo: ti.i32, hi_excl: ti.i32) -> ti.i32:
    return ti.max(lo, ti.min(v, hi_excl - 1))

@ti.data_oriented
class BaseContainer:
    def __init__(self, config: SimConfig, GGUI=False, gaussians=None, fluid_dir=None):
        self.cfg = config
        self.GGUI = GGUI
        self.total_time = 0.0

        
        self.fluid_bodies = self.cfg.get_fluid_bodies()
        self.fluid_blocks = self.cfg.get_fluid_blocks()
        assert len(self.fluid_blocks) <= 1, "Only support one fluid block for each simulation"
        fluid = self.fluid_blocks[0]  # only support one fluid block for now
        thick = fluid["thick"]

        # Material 0 indicates the object does not exist
        self.material_rigid = 2
        self.material_fluid = 1

        self.dx = 0.01  # particle radius
        self.dx = self.cfg.get_cfg("particleRadius")
        self.disable_rigid_particles = bool(self.cfg.get_cfg("disableRigidParticles") or False)

        self.particle_diameter = 2 * self.dx
        self.dim = len(fluid["translation"])
        if self.dim == 3:
            self.dh = self.dx * 4.0  # support radius
        else:
            self.dh = self.dx * 3.0  # support radius

        if self.cfg.get_cfg("supportRadius"):
            self.dh = self.cfg.get_cfg("supportRadius")
        

        self.particle_spacing = self.particle_diameter
        if self.cfg.get_cfg("particleSpacing"):
            self.particle_spacing = self.cfg.get_cfg("particleSpacing")


        self.grid_size = float(self.dh)  # ensure scalar float
        self.padding = float(self.grid_size)

        #### Process Rigid Bodies from Mesh ####
        rigid_body_particle_num = 0
        self.rigid_bodies = self.cfg.get_rigid_bodies()
        assert len(self.rigid_bodies) <= 1, "Only support one rigid body for each simulation"
        
        rigid_body = self.rigid_bodies[0]
        if rigid_body["geometryFile"] == "GS":
            self.gaussians = gaussians
            voxelized_points_np, transformation_params = self.load_gaussian_rigid(rigid_body, self.gaussians, pitch=self.particle_spacing, 
                                                            thresh=self.cfg.get_cfg("thresh_gaussian"), bbox_margin=self.cfg.get_cfg("bbox_margin"))
        elif rigid_body["geometryFile"] == "ply":
            pass
        else:
            voxelized_points_np = self.load_rigid_body(rigid_body, pitch=self.particle_spacing)
        rigid_body["particleNum"] = voxelized_points_np.shape[0]
        rigid_body["voxelizedPoints"] = voxelized_points_np
        rigid_body_particle_num += voxelized_points_np.shape[0]


        #### Process Rigid Blocks ####
        self.rigid_blocks = self.cfg.get_rigid_blocks()
        for rigid_block in self.rigid_blocks:
            raise NotImplementedError
            # particle_num = self.compute_cube_particle_num(rigid_block["start"], rigid_block["end"], space=self.particle_spacing)
            # rigid_block["particleNum"] = particle_num
            # rigid_body_particle_num += particle_num
        num_rigid_object = len(self.rigid_blocks) + len(self.rigid_bodies)
        # print(f"Number of rigid bodies and rigid blocks: {num_rigid_object}")   

        # ====================auto calculate domain range======================== #
        if self.cfg.get_cfg("range") == "auto":
            if fluid_dir is None:
                fluid_dir = [0, 1, 0] 
            padding = float(self.padding) * 5
            rmin = voxelized_points_np.min(axis=0).astype(float)
            rmax = voxelized_points_np.max(axis=0).astype(float)
            rlen = rmax - rmin
            a, sgn = _axis_from_dir(fluid_dir)
            fmin = rmin.copy()
            fmax = rmax.copy()

            if sgn > 0:
                fmin[a] = rmax[a] + padding
                fmax[a] = fmin[a] + thick
            else:
                fmax[a] = rmin[a] - padding
                fmin[a] = fmax[a] - thick

            axes = [0, 1, 2]
            others = [i for i in axes if i != a]
            if a == 1:
                pass
            else:
                oy, oz = (1, others[1]) if others[0] == 1 else (1, others[0])
                fmin[oy], fmax[oy] = _double_span(rmin[oy], rmax[oy])

            dmin = np.minimum(rmin, fmin)
            dmax = np.maximum(rmax, fmax)

            dmin -= padding * 5
            dmax += padding * 5

            need_len = rlen[a] + thick + padding
            cur_len = dmax[a] - dmin[a]
            if cur_len < need_len:
                deficit = need_len - cur_len
                if sgn > 0:
                    dmax[a] += deficit
                else:
                    dmin[a] -= deficit

            if a == 1 and sgn > 0:
                dmin[1] = min(dmin[1], rmin[1] - padding)  
                dmax[1] = 8.0  
            self.domain_start = dmin
            self.domain_end   = dmax

        else:
            self.domain_start = np.array([0.0, 0.0, 0.0])
            self.domain_start = np.array(self.cfg.get_cfg("domainStart"))
            assert self.domain_start[1] >= 0.0, "domain start y should be greater than 0"
            self.domain_end   = np.array([1.0, 1.0, 1.0])
            self.domain_end   = np.array(self.cfg.get_cfg("domainEnd"))

        self.domain_size = self.domain_end - self.domain_start
        

        self.V0 = 0.8 * self.particle_diameter ** self.dim
        self.particle_num = ti.field(dtype=ti.i32, shape=())
        self.particle_num[None] = 0

        self.max_num_object = 20

        # Grid related properties
        self.grid_num = tuple(int(v) for v in np.ceil(self.domain_size / self.grid_size).astype(int).tolist())
        assert all(g > 0 for g in self.grid_num), f"Bad grid_num {self.grid_num}"
        # print("grid size (cells): ", self.grid_num)

        self.grid_strides = []
        prod = 1
        for i in range(self.dim-1, -1, -1):
            self.grid_strides.insert(0, prod)
            prod *= self.grid_num[i]
        self.grid_strides = tuple(self.grid_strides)
        self.num_grid_flat = int(prod)

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()
        self.object_id_fluid_body = set()
        self.present_object = []


        margin = 0.03
        need = margin - self.domain_start
        shift = np.maximum(need, 0.0)  
        if np.any(shift > 0):
            self.domain_start = (self.domain_start + shift).astype(float)
            self.domain_end   = (self.domain_end   + shift).astype(float)
            rigid_body["voxelizedPoints"] = rigid_body["voxelizedPoints"] + shift
            voxelized_points_np = voxelized_points_np + shift
            self._global_shift = shift.astype(np.float32)
        else:
            self._global_shift = np.zeros_like(self.domain_start, dtype=np.float32)
        # self.domain_size = self.domain_end - self.domain_start
        ##########################################################################
        
        self.add_domain_box = self.cfg.get_cfg("addDomainBox")
        if self.add_domain_box:
            self.domain_box_start = [self.domain_start[i] + self.padding  for i in range(self.dim)]
            self.domain_box_end = [self.domain_end[i] - self.padding for i in range(self.dim)]
            self.domain_box_size = [self.domain_box_end[i] - self.domain_box_start[i] for i in range(self.dim)]
            self.domain_box_thickness = 0.03
        else:
            self.domain_box_thickness = 0.0

        #========== Compute number of particles ==========#
        #### Process Fluid Bodies from Mesh ####
        fluid_particle_num = 0

        for fluid_body in self.fluid_bodies:
            voxelized_points_np_fluid = self.load_fluid_body(fluid_body, pitch=self.particle_spacing)
            if hasattr(self, "_global_shift"):
                voxelized_points_np_fluid = voxelized_points_np_fluid + self._global_shift
            fluid_body["particleNum"] = voxelized_points_np_fluid.shape[0]
            fluid_body["voxelizedPoints"] = voxelized_points_np_fluid
            fluid_particle_num += voxelized_points_np_fluid.shape[0]
            # print("fluid max", voxelized_points_np_fluid.max(axis=0))
            # print("fluid min", voxelized_points_np_fluid.min(axis=0))

        #### Process Fluid Blocks ####
        if fluid["range"] == "auto":
            # padding = float(self.padding) * 5
            thick   = float(fluid.get("thick", self.cfg.get_cfg("fluidThickness") or 2.0))

            rmin = voxelized_points_np.min(axis=0).astype(float)
            rmax = voxelized_points_np.max(axis=0).astype(float)
            a, sgn = _axis_from_dir(fluid_dir)

            fstart = rmin.copy().tolist()
            fend   = rmax.copy().tolist()

            if sgn > 0:
                fstart[a] = (rmax[a] + padding)
                fend[a]   = fstart[a] + thick
            else:
                fend[a]   = (rmin[a] - padding)
                fstart[a] = fend[a] - thick

            if a == 1:  
                for ax in (0, 2):
                    fstart[ax] = rmin[ax]
                    fend[ax]   = rmax[ax]
            else:
                y0, y1 = _double_span(rmin[1], rmax[1])
                fstart[1] = y0
                fend[1]   = y1
                other_axis = [0, 2][0] if a == 2 else 2  
                fstart[other_axis] = rmin[other_axis]
                fend[other_axis]   = rmax[other_axis]
                
                fluid['velocity'] = - np.array(fluid_dir, dtype=float) * self.cfg.get_cfg("velocity_ratio", 1.0)

            fluid["start"] = fstart
            fluid["end"]   = fend
            

        # if fluid["range"] == "auto":
        #     fluid["start"] = voxelized_points_np.min(axis=0).tolist()
        #     fluid["end"] = voxelized_points_np.max(axis=0).tolist()
        #     fluid["start"][1] = fluid["end"][1] + self.padding # place above the rigid body
        #     fluid["end"][1] = fluid["start"][1] + thick
        # pdb.set_trace()
        particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"], space=self.particle_spacing)
        fluid["particleNum"] = particle_num
        fluid_particle_num += particle_num

        num_fluid_object = len(self.fluid_blocks) + len(self.fluid_bodies)

        self.fluid_particle_num = fluid_particle_num
        self.rigid_body_particle_num = rigid_body_particle_num
        self.particle_max_num = (
            fluid_particle_num 
            + rigid_body_particle_num 
            + (self.compute_box_particle_num(self.domain_box_start, self.domain_box_size, space=self.particle_spacing, thickness=self.domain_box_thickness) if self.add_domain_box else 0)
        )
        
        # print(f"Fluid particle num: {self.fluid_particle_num}, Rigid body particle num: {self.rigid_body_particle_num}")


        self.fluid_particle_num = ti.field(ti.i32, shape=())
        self.fluid_particle_num[None] = int(fluid_particle_num)


        #========== Allocate memory ==========#
        # Particle num of each grid
        num_grid = reduce(lambda x, y: x * y, self.grid_num) # handle 2d and 3d together
        
        self.grid_num_particles       = ti.field(dtype=ti.i32,  shape=self.num_grid_flat)
        self.grid_num_particles_temp  = ti.field(dtype=ti.i32,  shape=self.num_grid_flat)
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_num_particles.shape[0])

        self.particle_object_ids = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.particle_positions  = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.particle_velocities = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.particle_accelerations = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.particle_rest_volumes  = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_masses        = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_densities     = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_pressures     = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_materials     = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.particle_colors        = ti.Vector.field(3, dtype=ti.i32, shape=self.particle_max_num)
        self.particle_is_dynamic    = ti.field(dtype=ti.i32, shape=self.particle_max_num)

        self.object_materials       = ti.field(dtype=ti.i32, shape=self.max_num_object)
        self.object_num             = ti.field(dtype=ti.i32, shape=())
        self.object_num[None] = int(num_fluid_object + num_rigid_object + (1 if self.add_domain_box else 0))

        self.rigid_particle_original_positions = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.rigid_body_is_dynamic            = ti.field(dtype=ti.i32, shape=self.max_num_object)
        self.rigid_body_original_centers_of_mass = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_masses                = ti.field(dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_centers_of_mass       = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_rotations             = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_torques               = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_forces                = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_velocities            = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_angular_velocities    = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_num_object)
        self.rigid_body_particle_num          = ti.field(dtype=ti.i32, shape=self.max_num_object)

        # Buffers
        self.particle_object_ids_buffer = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.particle_positions_buffer  = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.rigid_particle_original_positions_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.particle_velocities_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.particle_rest_volumes_buffer = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_masses_buffer       = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_densities_buffer    = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_materials_buffer    = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.particle_colors_buffer       = ti.Vector.field(3, dtype=ti.i32, shape=self.particle_max_num)
        self.is_dynamic_buffer            = ti.field(dtype=ti.i32, shape=self.particle_max_num)

        self.object_visibility = ti.field(dtype=ti.i32, shape=self.max_num_object)

        self.grid_ids        = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.grid_ids_new    = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

        if self.add_domain_box:
            self.add_box(
                object_id=self.object_num[None]-1, # give the last object id to the domain box
                lower_corner=self.domain_box_start,
                cube_size=self.domain_box_size,
                thickness=self.domain_box_thickness,
                material=self.material_rigid,
                is_dynamic=False,
                space=self.particle_spacing,
                color=(127, 127, 127)
            )

            self.object_visibility[self.object_num[None]-1] = 0
            self.object_materials[self.object_num[None]-1] = self.material_rigid
            # self.object_id_rigid_body.add(self.object_num[None]-1)
            self.rigid_body_is_dynamic[self.object_num[None]-1] = 0
            self.rigid_body_velocities[self.object_num[None]-1] = ti.Vector([0.0 for _ in range(self.dim)])
            self.object_collection[self.object_num[None]-1] = 0 # dummy
        
        self.num_gauss = 0
        self.gauss_centers = None
        self.gauss_r2      = None
        self.gauss_sum_div = None
        self.gauss_count   = None
        self.gauss_sum_vor = None
        self.gauss_count_vor = None

        if gaussians is not None:
            means  = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)     # (N, 3 or dim)
            scales = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32) # (N, 3)
            means = (means - transformation_params['center']) @ transformation_params['R'].T + transformation_params['center']
            means += transformation_params['t']
            if hasattr(self, "_global_shift"):
                means = means + self._global_shift
            if means.ndim != 2 or means.shape[1] < self.dim:
                raise ValueError(f"means shape {means.shape} incompatible with dim={self.dim}")
            means_np = means[:, :self.dim].astype(np.float32, copy=False)             # (N, self.dim)

            # r = k * max(scales, axis=1)   -> r^2
            k = np.float32(3.0)
            r = (k * np.max(scales, axis=1)).astype(np.float32, copy=False)      
            r2_np = (r * r).astype(np.float32, copy=False)                        # (N,)
            # N = scales.shape[0]  
            # val = np.float32(k * float(self.dx))  
            # r2_np = np.full(N, val * val, dtype=np.float32)              # (N,)

            N = int(means_np.shape[0])
            self.num_gauss = N

            if N > 0:
                self.gauss_centers = ti.Vector.field(self.dim, dtype=ti.f32, shape=N)
                self.gauss_r2      = ti.field(dtype=ti.f32,  shape=N)
                self.gauss_sum_div = ti.field(dtype=ti.f32,  shape=N)
                self.gauss_count   = ti.field(dtype=ti.i32,  shape=N)
                self.gauss_sum_vor = ti.field(dtype=ti.f32,  shape=N)
                self.gauss_count_vor = ti.field(dtype=ti.i32, shape=N)
                self.gauss_sum_cfl = ti.field(dtype=ti.f32,  shape=N)
                self.gauss_count_cfl = ti.field(dtype=ti.i32, shape=N)
                
                # self.gauss_sum_energy_gap = ti.field(dtype=ti.f32, shape=N)
                # self.gauss_count_energy   = ti.field(dtype=ti.i32, shape=N)

                _centers_loaded = False
                try:
                    self.gauss_centers.from_numpy(means_np)
                    _centers_loaded = True
                except Exception:
                    _centers_loaded = False

                if not _centers_loaded:
                    centers_mat = ti.field(dtype=ti.f32, shape=(N, self.dim))
                    centers_mat.from_numpy(means_np)

                    @ti.kernel
                    def _pack_centers(mat: ti.template(), vec: ti.template()):
                        for i in range(N):
                            if self.dim == 3:
                                vec[i] = ti.Vector([mat[i, 0], mat[i, 1], mat[i, 2]])
                            elif self.dim == 2:
                                vec[i] = ti.Vector([mat[i, 0], mat[i, 1]])
                            else:
                                _v = ti.Vector.zero(ti.f32, self.dim)
                                for d in range(self.dim):
                                    _v[d] = mat[i, d]
                                vec[i] = _v
                    _pack_centers(centers_mat, self.gauss_centers)

                self.gauss_r2.from_numpy(r2_np)

                self.gauss_sum_div.fill(0.0)
                self.gauss_count.fill(0)
                self.gauss_sum_vor.fill(0.0)
                self.gauss_count_vor.fill(0)
                                
                self.gauss_sum_cfl.fill(0.0)
                self.gauss_count_cfl.fill(0)
                                
                # self.gauss_sum_energy_gap.fill(0.0) 
                # self.gauss_count_energy.fill(0) 

        self.particle_vorticity_mag = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.particle_velocity_divergence = ti.field(dtype=ti.f32, shape=self.particle_max_num) 
        self.particle_cfl = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        
        self.domain_start_ti = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        self.domain_end_ti   = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        self.padding_ti      = ti.field(dtype=ti.f32, shape=())

        self.domain_start_ti[None] = ti.Vector([float(x) for x in self.domain_start])
        self.domain_end_ti[None]   = ti.Vector([float(x) for x in self.domain_end])
        self.padding_ti[None]      = float(self.padding)

        self.particle_vel_smooth_vort = ti.Vector.field(self.dim, ti.f32, shape=self.particle_max_num)
        self.particle_vel_smooth_div  = ti.Vector.field(self.dim, ti.f32, shape=self.particle_max_num)

        # self.vel_smooth_radius = ti.field(dtype=ti.f32, shape=())
        # self.particle_vel_smooth = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.particle_need_vort = ti.field(dtype=ti.i32, shape=self.particle_max_num)





    def insert_object(self):
    ###### Add particles ######
        # Fluid block
        for fluid in self.fluid_blocks:
            obj_id = fluid["objectId"]

            if obj_id in self.present_object:
                continue
            if fluid["entryTime"] > self.total_time:
                continue
            # if hasattr(self, "_global_shift"):
            #     voxelized_points_np_fluid = voxelized_points_np_fluid + self._global_shift
            # print('self._global_shift', getattr(self, "_global_shift", None))
            offset = np.array(fluid["translation"])  # + self._global_shift  if hasattr(self, "_global_shift") else np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            # print(f"fluid block {obj_id} start {start}, end {end}, num {fluid['particleNum']}")
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.object_id_fluid_body.add(obj_id)

            if "visible" in fluid:
                self.object_visibility[obj_id] = fluid["visible"]
            else:
                self.object_visibility[obj_id] = 1

            self.object_materials[obj_id] = self.material_fluid
            self.object_collection[obj_id] = fluid
            

            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=1, 
                          color=color,
                          material=self.material_fluid,
                          space=self.particle_spacing)
            
            self.present_object.append(obj_id)

        # Fluid body
        for fluid_body in self.fluid_bodies:
            obj_id = fluid_body["objectId"]

            if obj_id in self.present_object:
                continue
            if fluid_body["entryTime"] > self.total_time:
                continue

            num_particles_obj = fluid_body["particleNum"]
            voxelized_points_np = fluid_body["voxelizedPoints"]
            velocity = np.array(fluid_body["velocity"], dtype=np.float32)
   
            density = fluid_body["density"]
            color = np.array(fluid_body["color"], dtype=np.int32)

            if "visible" in fluid_body:
                self.object_visibility[obj_id] = fluid_body["visible"]
            else:
                self.object_visibility[obj_id] = 1

            self.object_materials[obj_id] = self.material_fluid
            self.object_id_fluid_body.add(obj_id)
            self.object_collection[obj_id] = fluid_body

            self.add_particles(obj_id,
                                 num_particles_obj,
                                 np.array(voxelized_points_np, dtype=np.float32), # position
                                 np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                                 density * np.ones(num_particles_obj, dtype=np.float32), # density
                                 np.zeros(num_particles_obj, dtype=np.float32), # pressure
                                 np.array([self.material_fluid for _ in range(num_particles_obj)], dtype=np.int32), 
                                 1 * np.ones(num_particles_obj, dtype=np.int32), # dynamic
                                 np.stack([color for _ in range(num_particles_obj)]))

            self.present_object.append(obj_id)
            self.fluid_particle_num[None] += num_particles_obj

        # Rigid body
        for rigid_body in self.rigid_bodies:

            if self.disable_rigid_particles:
                obj_id = rigid_body["objectId"]
                self.object_visibility[obj_id] = 0
                continue
            obj_id = rigid_body["objectId"]

            if obj_id in self.present_object:
                continue
            if rigid_body["entryTime"] > self.total_time:
                continue

            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            self.rigid_body_particle_num[obj_id] = num_particles_obj
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body["color"], dtype=np.int32)

            if "visible" in rigid_body:
                self.object_visibility[obj_id] = rigid_body["visible"]
            else:
                self.object_visibility[obj_id] = 1

            self.object_materials[obj_id] = self.material_rigid
            self.object_collection[obj_id] = rigid_body

            #TODO: deal with different spacing
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.array([self.material_rigid for _ in range(num_particles_obj)], dtype=np.int32), 
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32), # is_dynamic
                               np.stack([color for _ in range(num_particles_obj)])) # color
        

            self.rigid_body_is_dynamic[obj_id] = is_dynamic
            self.rigid_body_velocities[obj_id] = velocity

            if is_dynamic:
                self.rigid_body_masses[obj_id] = self.compute_rigid_body_mass(obj_id)
                self.rigid_body_is_dynamic[obj_id] = 1
                # rigid_com = self.compute_rigid_body_center_of_mass(obj_id)
                # ! here we assume the center of mass is exactly the base frame center and calculated it in the bullet solver.
               
            self.present_object.append(obj_id)


        # Rigid block
        for rigid_block in self.rigid_blocks:
            raise NotImplementedError
            # obj_id = rigid_block["objectId"]

            # if obj_id in self.present_object:
            #     continue
            # if rigid_block["entryTime"] > self.total_time:
            #     continue

            # offset = np.array(rigid_block["translation"])
            # start = np.array(rigid_block["start"]) + offset
            # end = np.array(rigid_block["end"]) + offset
            # scale = np.array(rigid_block["scale"])
            # velocity = rigid_block["velocity"]
            # density = rigid_block["density"]
            # color = rigid_block["color"]
            # is_dynamic = rigid_block["isDynamic"]

            # if "visible" in rigid_block:
            #     self.object_visibility[obj_id] = rigid_block["visible"]
            # else:
            #     self.object_visibility[obj_id] = 1

            # self.object_materials[obj_id] = self.material_rigid

            # self.add_cube(object_id=obj_id,
            #               lower_corner=start,
            #               cube_size=(end-start)*scale,
            #               velocity=velocity,
            #               density=density, 
            #               is_dynamic=is_dynamic,
            #               color=color,
            #               material=self.material_rigid,
            #               space=self.particle_spacing) 
            # # TODO: compute center of mass and other information

            # self.present_object.append(obj_id)


    @ti.kernel
    def compute_rigid_body_mass(self, object_id: int) -> ti.f32:
        sum_m = 0.0
        for p_i in range(self.particle_num[None]):
            if self.particle_object_ids[p_i] == object_id and self.particle_is_dynamic[p_i]:
                sum_m += self.particle_densities[p_i] * self.V0
        return sum_m

    @ti.kernel
    def compute_rigid_body_center_of_mass(self, object_id: int) -> ti.types.vector(3, float):
        sum_xm = ti.Vector([0.0 for _ in range(self.dim)])
        sum_m = 0.0
        for p_i in range(self.particle_num[None]):
            if self.particle_object_ids[p_i] == object_id and self.particle_is_dynamic[p_i]:
                sum_xm += self.particle_positions[p_i] * self.particle_densities[p_i] * self.V0
                sum_m += self.particle_densities[p_i] * self.V0

        return sum_xm / sum_m

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.particle_object_ids[p] = obj_id
        self.particle_positions[p] = x
        self.rigid_particle_original_positions[p] = x
        self.particle_velocities[p] = v
        self.particle_densities[p] = density
        self.particle_rest_volumes[p] = self.V0
        self.particle_masses[p] = self.V0 * density
        self.particle_pressures[p] = pressure
        self.particle_materials[p] = material
        self.particle_is_dynamic[p] = is_dynamic
        self.particle_colors[p] = color
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):

        start = int(self.particle_num[None])
        end = start + int(new_particles_num)
        if end > int(self.particle_max_num):
            raise RuntimeError(f"add_particles overflow: {end} > particle_max_num={int(self.particle_max_num)}")
        

        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      )

    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num

    # ---------- kernels/helpers (3D only) ----------

    @ti.func
    def pos_to_index(self, pos):
        idx = ti.Vector.zero(ti.i32, self.dim)
        for d in ti.static(range(self.dim)):
            # gsz = ti.cast(self.grid_size, ti.f32)
            gsz = ti.max(ti.cast(self.grid_size, ti.f32), 1e-8)
            iv = ti.floor(pos[d] / gsz)
            iv_i32 = ti.cast(iv, ti.i32)
            gnum = ti.cast(self.grid_num[d], ti.i32)
            iv_i32 = ti.max(0, ti.min(iv_i32, gnum - 1))
            idx[d] = iv_i32
        return idx


    @ti.func
    def flatten_grid_index(self, grid_index):
        ret   = ti.cast(0, ti.i32)
        stride = ti.cast(1, ti.i32)
        for d in ti.static(range(self.dim - 1, -1, -1)):
            gi = ti.cast(grid_index[d], ti.i32)
            ret = ret + gi * stride
            stride = stride * ti.cast(self.grid_num[d], ti.i32)
        return ret


    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))


    # @ti.func
    # def pos_to_index(self, pos):
    #     # subtract domain_start and clamp per axis
    #     d0 = ti.static(float(self.domain_start[0]))
    #     d1 = ti.static(float(self.domain_start[1]))
    #     d2 = ti.static(float(self.domain_start[2]))
    #     nx = ti.static(self.grid_num[0])
    #     ny = ti.static(self.grid_num[1])
    #     nz = ti.static(self.grid_num[2])

    #     gx = ti.cast((pos[0] - d0) / self.grid_size, ti.i32)
    #     gy = ti.cast((pos[1] - d1) / self.grid_size, ti.i32)
    #     gz = ti.cast((pos[2] - d2) / self.grid_size, ti.i32)

    #     gx = _clampi(gx, 0, nx)
    #     gy = _clampi(gy, 0, ny)
    #     gz = _clampi(gz, 0, nz)
    #     return ti.Vector([gx, gy, gz])

    # @ti.func
    # def flatten_grid_index(self, grid_index):
    #     # row-major: x * (ny*nz) + y * nz + z
    #     s0 = ti.static(self.grid_strides[0])
    #     s1 = ti.static(self.grid_strides[1])
    #     s2 = ti.static(self.grid_strides[2])
    #     return grid_index[0] * s0 + grid_index[1] * s1 + grid_index[2] * s2

    # @ti.func
    # def get_flatten_grid_index(self, pos):
    #     return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.func
    def _flat_in_range(self, idx: ti.i32) -> ti.i32:
        nflat = ti.static(self.num_grid_flat)
        return ti.cast(0 <= idx and idx < nflat, ti.i32)


    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)


    @ti.func
    def flatten_grid_index(self, grid_index):
        ret = 0
        for i in ti.static(range(self.dim)):
            ret_p = grid_index[i]
            for j in ti.static(range(i+1, self.dim)):
                ret_p *= self.grid_num[j]
            ret += ret_p
        
        return ret
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))
    


    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.func
    def _grid_index_in_range(self, flat_idx: ti.i32) -> ti.i32:
        return ti.cast(0 <= flat_idx and flat_idx < self.num_grid_flat, ti.i32)

    @ti.func
    def is_static_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and (not self.particle_is_dynamic[p])

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and self.particle_is_dynamic[p]
    
    @ti.kernel
    def init_grid(self):
        self.grid_num_particles.fill(0)
        for p_i in range(self.particle_num[None]):
            grid_index = self.get_flatten_grid_index(self.particle_positions[p_i])
            self.grid_ids[p_i] = grid_index
            ti.atomic_add(self.grid_num_particles[grid_index], 1)
        for p_i in ti.grouped(self.grid_num_particles):
            self.grid_num_particles_temp[p_i] = self.grid_num_particles[p_i]
    
    @ti.kernel
    def reorder_particles(self):
        # ! if a property of a particle is not going to be directly recomputed in each iteration, it should be inside this function
        # ! debugging can be started from here
        
        for i in range(self.particle_num[None]):
            p_i = self.particle_num[None] - 1 - i
            base_offset = 0
            if self.grid_ids[p_i] - 1 >= 0:
                base_offset = self.grid_num_particles[self.grid_ids[p_i]-1]
            self.grid_ids_new[p_i] = ti.atomic_sub(self.grid_num_particles_temp[self.grid_ids[p_i]], 1) - 1 + base_offset

        for p_i in range(self.particle_num[None]):
            new_index = self.grid_ids_new[p_i]
            self.grid_ids_buffer[new_index] = self.grid_ids[p_i]
            self.particle_object_ids_buffer[new_index] = self.particle_object_ids[p_i]
            self.rigid_particle_original_positions_buffer[new_index] = self.rigid_particle_original_positions[p_i]
            self.particle_positions_buffer[new_index] = self.particle_positions[p_i]
            self.particle_velocities_buffer[new_index] = self.particle_velocities[p_i]
            self.particle_rest_volumes_buffer[new_index] = self.particle_rest_volumes[p_i]
            self.particle_masses_buffer[new_index] = self.particle_masses[p_i]
            self.particle_densities_buffer[new_index] = self.particle_densities[p_i]
            self.particle_materials_buffer[new_index] = self.particle_materials[p_i]
            self.particle_colors_buffer[new_index] = self.particle_colors[p_i]
            self.is_dynamic_buffer[new_index] = self.particle_is_dynamic[p_i]

        for p_i in range(self.particle_num[None]):
            self.grid_ids[p_i] = self.grid_ids_buffer[p_i]
            self.particle_object_ids[p_i] = self.particle_object_ids_buffer[p_i]
            self.rigid_particle_original_positions[p_i] = self.rigid_particle_original_positions_buffer[p_i]
            self.particle_positions[p_i] = self.particle_positions_buffer[p_i]
            self.particle_velocities[p_i] = self.particle_velocities_buffer[p_i]
            self.particle_rest_volumes[p_i] = self.particle_rest_volumes_buffer[p_i]
            self.particle_masses[p_i] = self.particle_masses_buffer[p_i]
            self.particle_densities[p_i] = self.particle_densities_buffer[p_i]
            self.particle_materials[p_i] = self.particle_materials_buffer[p_i]
            self.particle_colors[p_i] = self.particle_colors_buffer[p_i]
            self.particle_is_dynamic[p_i] = self.is_dynamic_buffer[p_i]

    def prepare_neighborhood_search(self):
        self.init_grid()
        self.prefix_sum_executor.run(self.grid_num_particles)
        self.reorder_particles()
    
    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        self.for_all_neighbors_radius(p_i, task, ret, -1.0)

    @ti.func
    def for_all_neighbors_radius(self, p_i, task: ti.template(), ret: ti.template(), radius: ti.f32):
        center_cell = self.pos_to_index(self.particle_positions[p_i])

        nx = ti.static(self.grid_num[0])
        ny = ti.static(self.grid_num[1])
        nz = ti.static(self.grid_num[2])

        # radius <= 0 => fallback to dh
        r = ti.select(radius > 0.0, radius, ti.cast(self.dh, ti.f32))
        r2 = r * r

        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            cell = center_cell + offset
            cell[0] = _clampi(cell[0], 0, nx)
            cell[1] = _clampi(cell[1], 0, ny)
            cell[2] = _clampi(cell[2], 0, nz)

            gidx = self.flatten_grid_index(cell)
            if self._grid_index_in_range(gidx) == 0:
                continue

            start_idx = 0
            end_idx = self.grid_num_particles[gidx]
            if gidx > 0:
                start_idx = self.grid_num_particles[gidx - 1]

            if start_idx < 0:
                start_idx = 0
            if end_idx > self.particle_num[None]:
                end_idx = self.particle_num[None]

            xi = self.particle_positions[p_i]
            for p_j in range(start_idx, end_idx):
                if p_i != p_j:
                    dx = xi - self.particle_positions[p_j]
                    if dx.dot(dx) < r2:
                        task(p_i, p_j, ret)



    # @ti.func
    # def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
    #     center_cell = self.pos_to_index(self.particle_positions[p_i])

    #     nx = ti.static(self.grid_num[0])
    #     ny = ti.static(self.grid_num[1])
    #     nz = ti.static(self.grid_num[2])

    #     for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
    #         cell = center_cell + offset
    #         # clamp per axis
    #         cell[0] = _clampi(cell[0], 0, nx)
    #         cell[1] = _clampi(cell[1], 0, ny)
    #         cell[2] = _clampi(cell[2], 0, nz)

    #         gidx = self.flatten_grid_index(cell)
    #         if self._grid_index_in_range(gidx) == 0:
    #             continue

    #         start_idx = 0
    #         end_idx = self.grid_num_particles[gidx]
    #         if gidx > 0:
    #             start_idx = self.grid_num_particles[gidx - 1]
    #         # clamp range for safety
    #         if start_idx < 0:
    #             start_idx = 0
    #         if end_idx > self.particle_num[None]:
    #             end_idx = self.particle_num[None]

    #         for p_j in range(start_idx, end_idx):
    #             if p_i != p_j and (self.particle_positions[p_i] - self.particle_positions[p_j]).norm() < self.dh:
    #                 task(p_i, p_j, ret)

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, invisible_objects=[], dim=3):
        self.flush_vis_buffer()
        for obj_id in self.object_collection:
            if self.object_visibility[obj_id] == 1:
                if dim ==3:
                    self._copy_to_vis_buffer_3d(obj_id)
                elif dim == 2:
                    self._copy_to_vis_buffer_2d(obj_id)

    @ti.kernel
    def flush_vis_buffer(self):
        self.x_vis_buffer.fill(0.0)
        self.color_vis_buffer.fill(0.0)

    @ti.kernel
    def _copy_to_vis_buffer_2d(self, obj_id: int):
        assert self.GGUI
        domain_size = ti.Vector([self.domain_size[0], self.domain_size[1]])
        for i in range(self.particle_max_num):
            if self.particle_object_ids[i] == obj_id:
                self.x_vis_buffer[i] = self.particle_positions[i] / domain_size
                self.color_vis_buffer[i] = self.particle_colors[i] / 255.0

    @ti.kernel
    def _copy_to_vis_buffer_3d(self, obj_id: int):
        assert self.GGUI
        # FIXME: make it equal to actual particle num
        for i in range(self.particle_max_num):
            if self.particle_object_ids[i] == obj_id:
                self.x_vis_buffer[i] = self.particle_positions[i]
                self.color_vis_buffer[i] = self.particle_colors[i] / 255.0

    def dump(self, obj_id):
        np_object_id = self.particle_object_ids.to_numpy()
        mask = (np_object_id == obj_id).nonzero()

        np_x = self.particle_positions.to_numpy()[mask]
        np_v = self.particle_velocities.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }
    
    def load_rigid_body(self, rigid_body, pitch=None):
        if pitch is None:
            pitch = self.particle_diameter
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])

        if rigid_body["isDynamic"] == False:
            # for static rigid body, we will not run renew_rigid_particle_state function. So we put them in the right place here
            offset = np.array(rigid_body["translation"])
            angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
            direction = rigid_body["rotationAxis"]
            rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
            mesh.apply_transform(rot_matrix)
            mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        rigid_body["mesh"] = mesh_backup
        rigid_body["restPosition"] = mesh_backup.vertices
        rigid_body["restCenterOfMass"] = np.array([0.0, 0.0, 0.0]) # ! if the center of mass is not exactly the base frame center, this will lead to error
        is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)

        voxelized_mesh = mesh.voxelized(pitch=pitch)
        voxelized_mesh = mesh.voxelized(pitch=pitch).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        voxelized_points_np = voxelized_mesh.points
        # print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")

        # voxelized_points_np = tm.sample.sample_surface_even(mesh, 4000)[0]
        
        return voxelized_points_np

    def load_gaussian_rigid(self, rigid_body,gaussians, pitch=None, thresh=0.4, bbox_margin=3.0):
        if pitch is None:
            pitch = self.particle_diameter

        if hasattr(gaussians, "get_xyz"):  # GaussianModel
            means   = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)          # (N,3)
            scales  = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)      # (N,3) activated scale
            rotations   = gaussians.get_rotation.detach().cpu().numpy().astype(np.float32)     # (N,4) normalized wxyz

            if gaussians.get_opacity is not None and gaussians.get_opacity.numel() > 0:
                weights = gaussians.get_opacity.detach().cpu().numpy().astype(np.float32) # .reshape(-1)  # (N,)
            else:
                weights = None
            print("weights",weights.min(), weights.max(), weights.mean())
        else:
            # dict: {means, scales, rotations|quaternions, weights? , log_scale?, quat_order?}
            means  = np.asarray(gaussians["means"], dtype=np.float32)
            scales = np.asarray(gaussians["scales"], dtype=np.float32)
            if gaussians.get("log_scale", False):
                scales = np.exp(scales).astype(np.float32)
            if "rotations" in gaussians:
                rotations = np.asarray(gaussians["rotations"], dtype=np.float32)
            else:
                rotations = np.asarray(gaussians["quaternions"], dtype=np.float32)
            weights = None
            if "weights" in gaussians:  # or opacity
                weights = np.asarray(gaussians["weights"], dtype=np.float32)
            elif "opacity" in gaussians:
                weights = np.asarray(gaussians["opacity"], dtype=np.float32)

        # print("transforming gaussian to voxel points...")
        voxelized_points_np = _gs_to_voxel_points(
            means, scales, rotations, weights=weights,
            voxel_size=float(pitch),
            bbox_margin=float(bbox_margin),
            thresh=float(thresh)
        )

        scene_scale = np.array(rigid_body.get("scale", [1,1,1]), dtype=np.float32)
        if not np.allclose(scene_scale, 1.0):
            voxelized_points_np = voxelized_points_np * scene_scale

        if rigid_body.get("isDynamic", True) == False:
            offset = np.array(rigid_body["translation"], dtype=np.float32)
            angle = rigid_body["rotationAngle"] / 360.0 * 2.0 * np.pi
            axis = np.array(rigid_body["rotationAxis"], dtype=np.float32)
            axis = axis / (np.linalg.norm(axis)+1e-8)
            K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]], dtype=np.float32)
            R_scene = np.eye(3, dtype=np.float32) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
            center = voxelized_points_np.mean(axis=0)
            voxelized_points_np = (voxelized_points_np - center) @ R_scene.T + center
            voxelized_points_np += offset
            transformation_params = {
                "R": R_scene,
                "t": offset,
                "center": center
            }

        rigid_body["mesh"] = None
        rigid_body["restPosition"] = voxelized_points_np.copy()
        rigid_body["restCenterOfMass"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # print(f"rigid body {rigid_body['objectId']} (gaussian) num: {voxelized_points_np.shape[0]}")

        # print(f"max extent: {voxelized_points_np.max(axis=0)}, min extent: {voxelized_points_np.min(axis=0)}")
        if voxelized_points_np.shape[0] <=200:
            import pdb; pdb.set_trace()
            raise RuntimeError(f"Gaussian rigid body {rigid_body['objectId']} has zero particles after voxelization! Please adjust the threshold or check the gaussian parameters.")
        return voxelized_points_np, transformation_params

        # # if you need to fill the mesh with evenly spaced particles, use the following code
        # # this piece of code is also used to create fluid object from mesh
        # min_point, max_point = mesh.bounding_box.bounds
        # num_dim = []
        # for i in range(self.dim):
        #     num_dim.append(
        #         np.arange(min_point[i], max_point[i], pitch))
        
        # new_positions = np.array(np.meshgrid(*num_dim,
        #                                      sparse=False,
        #                                      indexing='ij'),
        #                          dtype=np.float32)
        # new_positions = new_positions.reshape(-1,
        #                                       reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        # print(f"processing {len(new_positions)} points to decide whether they are inside the mesh. This might take a while.")
        # inside = [False for _ in range(len(new_positions))]

        # # decide whether the points are inside the mesh or not
        # # TODO: make it parallel or precompute and store
        # pbar = tqdm(total=len(new_positions))
        # for i in range(len(new_positions)):
        #     if mesh.contains([new_positions[i]])[0]:
        #         inside[i] = True
        #     pbar.update(1)

        # pbar.close()

        # new_positions = new_positions[inside]
        # return new_positions

    def load_fluid_body(self, rigid_body, pitch=None):
        if pitch is None:
            pitch = self.particle_diameter
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset

        min_point, max_point = mesh.bounding_box.bounds
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(min_point[i], max_point[i], pitch))
        
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        # print(f"processing {len(new_positions)} points to decide whether they are inside the mesh. This might take a while.")
        inside = [False for _ in range(len(new_positions))]

        # decide whether the points are inside the mesh or not
        # TODO: make it parallel or precompute and store
        pbar = tqdm(total=len(new_positions))
        for i in range(len(new_positions)):
            if mesh.contains([new_positions[i]])[0]:
                inside[i] = True
            pbar.update(1)

        pbar.close()

        new_positions = new_positions[inside]
        return new_positions

    def compute_cube_particle_num(self, start, end, space=None):
        if space is None:
            space = self.particle_diameter
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], space))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])

    def compute_box_particle_num(self, lower_corner, cube_size, thickness, space=None):
        if space is None:
            space = self.particle_diameter
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          space))
            
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        # remove particles inside the box
        # create mask
        mask = np.zeros(new_positions.shape[0], dtype=bool)
        for i in range(self.dim):
            mask = mask | ((new_positions[:, i] <= lower_corner[i] + thickness) | (new_positions[:, i] >= lower_corner[i] + cube_size[i] - thickness))
        new_positions = new_positions[mask]
        return new_positions.shape[0]

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 is_dynamic,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None,
                 space=None,
                ):
        """
        add particles spaced by space in a cube
        """
        if space is None:
            space = self.particle_diameter
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          space))
            
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()

        num_new_particles = new_positions.shape[0]

        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)

        if material == self.material_fluid:
            self.fluid_particle_num[None] += num_new_particles

    def add_box(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 thickness,
                 material,
                 is_dynamic,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None,
                 space=None,
                ):
        if space is None:
            space = self.particle_diameter
        num_dim = []
        
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          space))
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        # remove particles inside the box
        # create mask
        mask = np.zeros(new_positions.shape[0], dtype=bool)

        for i in range(self.dim):
            mask = mask | ((new_positions[:, i] <= lower_corner[i] + thickness) | (new_positions[:, i] >= lower_corner[i] + cube_size[i] - thickness))
            #! for testing
            # mask = mask | (new_positions[:, i] <= lower_corner[i] + thickness)
        new_positions = new_positions[mask]

        num_new_particles = new_positions.shape[0]

        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)

        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)

