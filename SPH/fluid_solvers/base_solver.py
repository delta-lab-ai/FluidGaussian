# implicit viscosity solver implemented as paper "A Physically Consistent Implicit Viscosity Solver for SPH Fluids"
import taichi as ti
import numpy as np
from ..containers import BaseContainer
from ..rigid_solver import PyBulletSolver

@ti.data_oriented
class BaseSolver():
    def __init__(self, container: BaseContainer):
        self.container = container
        self.cfg = container.cfg
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.container.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        
        self.g = np.array(self.container.cfg.get_cfg("gravitation"))

        # this is used to realize emitter. If a fluid particle is above this height,
        # we shall make it a rigid particle and fix its speed.
        # it's an awful hack, but it works. feel free to change it.
        self.g_upper = self.container.cfg.get_cfg("gravitationUpper")
        if self.g_upper == None:
            self.g_upper = 10000.0 # a large number

        self.viscosity_method = self.container.cfg.get_cfg("viscosityMethod")
        self.viscosity = self.container.cfg.get_cfg("viscosity")
        self.viscosity_b = self.container.cfg.get_cfg("viscosity_b")
        if self.viscosity_b == None:
            self.viscosity_b = self.viscosity
        self.density_0 = 1000.0  
        self.density_0 = self.container.cfg.get_cfg("density0")
        self.surface_tension = 0.01

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

        self.rigid_solver = PyBulletSolver(container, gravity=self.g,  dt=self.dt[None])

        if self.viscosity_method == "implicit":
            # initialize things needed for conjugate gradient solver
            # conjugate gradient solver implemented following https://en.wikipedia.org/wiki/Conjugate_gradient_method
            self.cg_p = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.original_velocity = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_Ap = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_x = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_b = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_alpha = ti.field(dtype=ti.f32, shape=())
            self.cg_beta = ti.field(dtype=ti.f32, shape=())
            self.cg_r = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_error = ti.field(dtype=ti.f32, shape=())
            self.cg_diagnol_ii_inv = ti.Matrix.field(self.container.dim, self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)

            self.cg_tol = 1e-6

        self.gravity_g = ti.field(dtype=ti.f32, shape=())
        self.vertical_axis = ti.field(dtype=ti.i32, shape=())  # 0:x, 1:y, 2:z

        # self.initial_particle_energy = ti.field(dtype=ti.f32, shape=())

        self._e_sum_h  = ti.field(dtype=ti.f32, shape=())
        self._e_sum_v2 = ti.field(dtype=ti.f32, shape=())
        self._e_count  = ti.field(dtype=ti.i32,  shape=())
        self._e_sum_h[None] = 0.0
        self._e_sum_v2[None] = 0.0
        self._e_count[None] = 0

    # -------------------------- CFL per-particle -----------------------------------
    @ti.kernel
    def compute_cfl_number(self, dt: ti.f32):
        for p in range(self.container.particle_num[None]):
            if self.container.particle_materials[p] == self.container.material_fluid:
                v = self.container.particle_velocities[p]
                vmag = v.norm()
                cfl = 0.0
                if self.container.dx > 0.0:          # avoid division by zero
                    cfl = vmag * dt / self.container.dx
                self.container.particle_cfl[p] = cfl
            else:
                self.container.particle_cfl[p] = 0.0

    @ti.kernel
    def accumulate_cfl_around_gaussians(self, N_g: int):
        for p in range(int(self.container.particle_num[None])):
            if self.container.particle_materials[p] == self.container.material_fluid:
                x   = self.container.particle_positions[p]
                cfl = self.container.particle_cfl[p]
                for g in range(N_g):
                    dx = x - self.container.gauss_centers[g]
                    if dx.dot(dx) <= self.container.gauss_r2[g]:
                        ti.atomic_add(self.container.gauss_sum_cfl[g],  cfl)
                        ti.atomic_add(self.container.gauss_count_cfl[g], 1)

    @ti.kernel
    def reduce_max_cfl(self):
        self.container.max_cfl[None] = 0.0
        for p in range(self.container.particle_num[None]):
            if self.container.particle_materials[p] == self.container.material_fluid:
                c = self.container.particle_cfl[p]
                if c > self.container.max_cfl[None]:
                    self.container.max_cfl[None] = c



    # -------------------------- vortex ----------------------------------------------
    @ti.func
    def _vor_task(self, p_i, p_j, acc: ti.template()):
        # SPH: sum_j V_j * ((v_j - v_i) x ∇W_ij)
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        x_i = self.container.particle_positions[p_i]
        x_j = self.container.particle_positions[p_j]
        grad = self.kernel_gradient(x_i - x_j)
        Vij  = self.container.particle_rest_volumes[p_j]

        if ti.static(self.container.dim == 3):
            acc += Vij * ti.math.cross(v_j - v_i, grad)
        else:
            acc += Vij * ((v_j - v_i)[0] * grad[1] - (v_j - v_i)[1] * grad[0])

    @ti.kernel
    def compute_vorticity(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                acc = ti.Vector.zero(ti.f32, 3)
                self.container.for_all_neighbors(p_i, self._vor_task, acc)
                self.container.particle_vorticity_mag[p_i] = acc.norm()
            else:
                self.container.particle_vorticity_mag[p_i] = 0.0

    # -------------------------- uniform vorticity (normalized kernel average) --------------------------
    @ti.func
    def _vel_smooth_task(self, p_i, p_j, ret: ti.template()):
        # ret: length (dim+1)
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            v_j = self.container.particle_velocities[p_j]
            Vj  = self.container.particle_rest_volumes[p_j]

            for d in ti.static(range(self.container.dim)):
                ret[d] += Vj * v_j[d]
            ret[self.container.dim] += Vj

    @ti.kernel
    def smooth_velocity_for_vorticity(self, radius: ti.f32):
        for p in range(self.container.particle_num[None]):
            self.container.particle_vel_smooth_vort[p] = self.container.particle_velocities[p]

        # R = self.container.vel_smooth_radius[None]  # <=0 falls back to dh (handled by for_all_neighbors)

        for p_i in range(self.container.particle_num[None]):
            if (self.container.particle_materials[p_i] == self.container.material_fluid and
                self.container.particle_need_vort[p_i] == 1):

                ret = ti.Vector.zero(ti.f32, self.container.dim + 1)

                # include self
                vi = self.container.particle_velocities[p_i]
                Vi = self.container.particle_rest_volumes[p_i]
                for d in ti.static(range(self.container.dim)):
                    ret[d] += Vi * vi[d]
                ret[self.container.dim] += Vi

                self.container.for_all_neighbors_radius(p_i, self._vel_smooth_task, ret, radius)

                sum_w = ret[self.container.dim]
                vsm = ti.Vector.zero(ti.f32, self.container.dim)
                for d in ti.static(range(self.container.dim)):
                    vsm[d] = ret[d] / (sum_w + 1e-8)

                self.container.particle_vel_smooth_vort[p_i] = vsm

    @ti.func
    def _u_vor_task(self, p_i, p_j, acc: ti.template()):
        v_i = self.container.particle_vel_smooth_vort[p_i]
        v_j = self.container.particle_vel_smooth_vort[p_j]

        x_i = self.container.particle_positions[p_i]
        x_j = self.container.particle_positions[p_j]
        grad = self.kernel_gradient(x_i - x_j)
        Vij  = self.container.particle_rest_volumes[p_j]

        if ti.static(self.container.dim == 3):
            acc += Vij * ti.math.cross(v_j - v_i, grad)
        else:
            acc += Vij * ((v_j - v_i)[0] * grad[1] - (v_j - v_i)[1] * grad[0])


    @ti.kernel
    def compute_uniform_vorticity(self):
        for p_i in range(self.container.particle_num[None]):
            if (self.container.particle_materials[p_i] == self.container.material_fluid and
                self.container.particle_need_vort[p_i] == 1):

                acc = ti.Vector.zero(ti.f32, 3 if ti.static(self.container.dim == 3) else 2)
                self.container.for_all_neighbors(p_i, self._u_vor_task, acc)  # -1 => dh
                self.container.particle_vorticity_mag[p_i] = acc.norm()
            else:
                self.container.particle_vorticity_mag[p_i] = 0.0


    @ti.kernel
    def accumulate_vorticity_around_gaussians(self, N_g: int):
        for p in range(int(self.container.particle_num[None])):
            if self.container.particle_materials[p] == self.container.material_fluid:
                x = self.container.particle_positions[p]
                wmag = self.container.particle_vorticity_mag[p]
                for g in range(N_g):
                    dx = x - self.container.gauss_centers[g]
                    if dx.dot(dx) <= self.container.gauss_r2[g]:
                        ti.atomic_add(self.container.gauss_sum_vor[g], wmag)
                        ti.atomic_add(self.container.gauss_count_vor[g], 1)

    # -------------------------- divergence ----------------------------------------------
    @ti.func
    def _div_task(self, p_i, p_j, acc: ti.template()):
        # SPH approximation: sum_j V_j * (v_j - v_i) · ∇W(x_i - x_j)
        v_i = self.container.particle_velocities[p_i]
        v_j = self.container.particle_velocities[p_j]
        x_i = self.container.particle_positions[p_i]
        x_j = self.container.particle_positions[p_j]
        acc += self.container.particle_rest_volumes[p_j] * ti.math.dot(v_j - v_i, self.kernel_gradient(x_i - x_j))


    @ti.kernel
    def compute_velocity_divergence(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                acc = 0.0
                self.container.for_all_neighbors(p_i, self._div_task, acc)
                self.container.particle_velocity_divergence[p_i] = ti.abs(acc)
            else:
                self.container.particle_velocity_divergence[p_i] = 0.0

    @ti.kernel
    def accumulate_divergence_around_gaussians(self, N_g: int):

        for p in range(int(self.container.particle_num[None])):
            if self.container.particle_materials[p] == self.container.material_fluid:
                x = self.container.particle_positions[p]
                d = self.container.particle_velocity_divergence[p]
                for g in range(N_g): 
                    dx = x - self.container.gauss_centers[g]
                    if dx.dot(dx) <= self.container.gauss_r2[g]:
                        ti.atomic_add(self.container.gauss_sum_div[g], d)
                        ti.atomic_add(self.container.gauss_count[g], 1)

    # -------------------------- uniform divergence (use smoothed velocity) --------------------------
    @ti.func
    def _u_div_task(self, p_i, p_j, acc: ti.template()):
        # SPH approximation on smoothed velocity:
        # sum_j V_j * (v_j^s - v_i^s) · ∇W(x_i - x_j)
        v_i = self.container.particle_vel_smooth_div[p_i]
        v_j = self.container.particle_vel_smooth_div[p_j]
        x_i = self.container.particle_positions[p_i]
        x_j = self.container.particle_positions[p_j]
        acc += self.container.particle_rest_volumes[p_j] * ti.math.dot(
            v_j - v_i, self.kernel_gradient(x_i - x_j)
        )

    @ti.kernel
    def compute_uniform_divergence(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                acc = 0.0
                self.container.for_all_neighbors(p_i, self._u_div_task, acc)  # -1 => dh
                self.container.particle_velocity_divergence[p_i] = ti.abs(acc)
            else:
                self.container.particle_velocity_divergence[p_i] = 0.0


    @ti.kernel
    def smooth_velocity_for_divergence(self, radius: ti.f32):
        # init: default to original velocity
        for p in range(self.container.particle_num[None]):
            self.container.particle_vel_smooth_div[p] = self.container.particle_velocities[p]

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector.zero(ti.f32, self.container.dim + 1)

                # include self
                vi = self.container.particle_velocities[p_i]
                Vi = self.container.particle_rest_volumes[p_i]
                for d in ti.static(range(self.container.dim)):
                    ret[d] += Vi * vi[d]
                ret[self.container.dim] += Vi

                # neighbor accumulation (same task as vorticity smoothing)
                self.container.for_all_neighbors_radius(p_i, self._vel_smooth_task, ret, radius)

                sum_w = ret[self.container.dim]
                vsm = ti.Vector.zero(ti.f32, self.container.dim)
                for d in ti.static(range(self.container.dim)):
                    vsm[d] = ret[d] / (sum_w + 1e-8)

                self.container.particle_vel_smooth_div[p_i] = vsm



    @ti.func
    def kernel_W(self, R_mod):
        # cubic kernel
        res = ti.cast(0.0, ti.f32)
        h = self.container.dh
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.container.dim == 1:
            k = 4 / 3
        elif self.container.dim == 2:
            k = 40 / 7 / np.pi
        elif self.container.dim == 3:
            k = 8 / np.pi
        k /= h ** self.container.dim
        q = R_mod / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def kernel_gradient(self, R):
        # cubic kernel gradient
        h = self.container.dh
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.container.dim == 1:
            k = 4 / 3
        elif self.container.dim == 2:
            k = 40 / 7 / np.pi
        elif self.container.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.container.dim
        R_mod = R.norm()
        q = R_mod / h
        res = ti.Vector([0.0 for _ in range(self.container.dim)])
        if R_mod > 1e-5 and q <= 1.0:
            grad_q = R / (R_mod * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.kernel
    def compute_rigid_particle_volume(self):
        # implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_positions[p_i][1] <= self.g_upper:
                    ret = self.kernel_W(0.0)
                    self.container.for_all_neighbors(p_i, self.compute_rigid_particle_volumn_task, ret)
                    self.container.particle_rest_volumes[p_i] = 1.0 / ret 
                    self.container.particle_masses[p_i] = self.density_0 * self.container.particle_rest_volumes[p_i]

    @ti.func
    def compute_rigid_particle_volumn_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_object_ids[p_j] == self.container.particle_object_ids[p_i]:
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.kernel_W(R_mod)

    @ti.kernel
    def init_acceleration(self):
        self.container.particle_accelerations.fill(0.0)

    @ti.kernel
    def init_rigid_body_force_and_torque(self):
        self.container.rigid_body_forces.fill(0.0)
        self.container.rigid_body_torques.fill(0.0)


    @ti.kernel
    def compute_pressure_acceleration(self):
        self.container.particle_accelerations.fill(0.0)
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_is_dynamic[p_i]:
                self.container.particle_accelerations[p_i] = ti.Vector([0.0 for _ in range(self.container.dim)])
                if self.container.particle_materials[p_i] == self.container.material_fluid:
                    ret_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                    self.container.for_all_neighbors(p_i, self.compute_pressure_acceleration_task, ret_i)
                    self.container.particle_accelerations[p_i] = ret_i

    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        # compute pressure acceleration from the gradient of pressure
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            den_j = self.container.particle_densities[p_j]

            ret += (
                - self.container.particle_masses[p_j] 
                * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_j] / (den_j * den_j)) 
                * nabla_ij
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # use fluid particle pressure, density as rigid particle pressure, density
            den_j = self.density_0
            acc = (
                - self.density_0 * self.container.particle_rest_volumes[p_j] 
                * self.container.particle_pressures[p_i] / (den_i * den_i)
                * nabla_ij
            )
            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                # add force and torque to rigid body
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                force_j = (
                    self.density_0 * self.container.particle_rest_volumes[p_j] 
                    * self.container.particle_pressures[p_i] / (den_i * den_i)
                    * nabla_ij
                    * (self.density_0 * self.container.particle_rest_volumes[p_i])
                )

                torque_j = ti.math.cross(pos_i - center_of_mass_j, force_j)
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += torque_j


    def compute_non_pressure_acceleration(self):
        # compute acceleration from gravity, surface tension and viscosity
        self.compute_gravity_acceleration()
        self.compute_surface_tension_acceleration()

        if self.viscosity_method == "standard":
            self.compute_viscosity_acceleration_standard()
        elif self.viscosity_method == "implicit":
            self.implicit_viscosity_solve()
        else:
            raise NotImplementedError(f"viscosity method {self.viscosity_method} not implemented")
        
    @ti.kernel
    def compute_gravity_acceleration(self):
        # assign g to all fluid particles, not +=
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_accelerations[p_i] =  ti.Vector(self.g)

    @ti.kernel
    def compute_surface_tension_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_surface_tension_acceleration_task, a_i)
                self.container.particle_accelerations[p_i] += a_i

    @ti.func
    def compute_surface_tension_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            diameter2 = self.container.particle_diameter * self.container.particle_diameter
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R2 = ti.math.dot(R, R)
            if R2 > diameter2:
                ret -= self.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.kernel_W(R.norm())
            else:
                ret -= self.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.kernel_W(ti.Vector([self.container.particle_diameter, 0.0, 0.0]).norm())            

    @ti.kernel
    def compute_viscosity_acceleration_standard(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_viscosity_acceleration_standard_task, a_i)
                self.container.particle_accelerations[p_i] += (a_i / self.density_0)

    @ti.func
    def compute_viscosity_acceleration_standard_task(self, p_i, p_j, ret: ti.template()):
        # we leave / self.density_0 to the caller. so we should do this when we compute rigid torque and force
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        # Compute the viscosity force contribution
        R = pos_i - pos_j
        nabla_ij = self.kernel_gradient(R)
        v_xy = ti.math.dot(self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j], R)
        
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            m_ij = (self.container.particle_masses[p_i] + self.container.particle_masses[p_j]) / 2
            acc = (
                2 * (self.container.dim + 2) * self.viscosity * m_ij 
                / self.container.particle_densities[p_j]
                / (R.norm()**2 + 0.01 * self.container.dh**2)  
                * v_xy 
                * nabla_ij
            )
            ret += acc

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            m_ij = (self.density_0 * self.container.particle_rest_volumes[p_j])
            acc = (
                2 * (self.container.dim + 2) * self.viscosity_b * m_ij
                / self.container.particle_densities[p_i]
                / (R.norm()**2 + 0.01 * self.container.dh**2)
                * v_xy
                * nabla_ij
            )

            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                force_j =  - acc * self.container.particle_masses[p_i] / self.density_0
                torque_j = ti.math.cross(pos_j - center_of_mass_j, force_j)
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += torque_j
    
    ################# Implicit Viscosity Solver #################
    @ti.kernel
    def prepare_conjugate_gradient_solver1(self):
        # initialize conjugate gradient solver phase 1
        self.cg_r.fill(0.0)
        self.cg_p.fill(0.0)
        self.original_velocity.fill(0.0)
        self.cg_b.fill(0.0)
        self.cg_Ap.fill(0.0)

        # initial guess for x. We use v^{df} + v(t) - v^{df}(t - dt) as initial guess. we assume v(t) - v^{df}(t - dt) is already in x
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_x[p_i] += self.container.particle_velocities[p_i]

        # storing the original velocity
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.original_velocity[p_i] = self.container.particle_velocities[p_i]

        # prepare b
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Matrix.zero(ti.f32, self.container.dim, self.container.dim)
                self.container.for_all_neighbors(p_i, self.compute_A_ii_task, ret)

                # preconditioner
                diag_ii = ti.Matrix.identity(ti.f32, self.container.dim) - ret * self.dt[None] / self.density_0
                self.cg_diagnol_ii_inv[p_i] = ti.math.inverse(diag_ii)

                ret1 = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_b_i_task, ret1)
                self.cg_b[p_i] = self.container.particle_velocities[p_i] - self.dt[None] * ret1 / self.density_0

                # copy x into p to calculate Ax
                self.cg_p[p_i] = self.cg_x[p_i]

    @ti.kernel
    def prepare_conjugate_gradient_solver2(self):
        # initialize conjugate gradient solver phase 2
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_r[p_i] = self.cg_diagnol_ii_inv[p_i]@self.cg_b[p_i] - self.cg_Ap[p_i]
                self.cg_p[p_i] = self.cg_r[p_i]

    @ti.func
    def compute_A_ii_task(self, p_i, p_j, ret: ti.template()):
        # there is left densities[p_i] to be divided
        # we assume p_i is a fluid particle
        # p_j can either be a fluid particle or a rigid particle
        A_ij = self.compute_A_ij(p_i, p_j)
        ret -= A_ij

    @ti.func
    def compute_b_i_task(self, p_i, p_j, ret: ti.template()):
        # we assume p_i is a fluid particle
        if self.container.particle_materials[p_j] == self.container.material_rigid:
            R = self.container.particle_positions[p_i] - self.container.particle_positions[p_j]
            nabla_ij = self.kernel_gradient(R)
            ret += (
                2 * (self.container.dim + 2) * self.viscosity_b
                * self.density_0 * self.container.particle_rest_volumes[p_j]
                / self.container.particle_densities[p_i]
                * ti.math.dot(self.container.particle_velocities[p_j], R)
                / (R.norm_sqr() + 0.01 * self.container.dh**2)
                * nabla_ij
            )
    
    @ti.func
    def compute_A_ij(self, p_i, p_j):
        # we do not divide densities[p_i] here.
        # we assume p_i is a fluid particle
        A_ij = ti.Matrix.zero(ti.f32, self.container.dim, self.container.dim)
        R = self.container.particle_positions[p_i] - self.container.particle_positions[p_j]
        nabla_ij = self.kernel_gradient(R)
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            m_ij = (self.container.particle_masses[p_i] + self.container.particle_masses[p_j]) / 2
            A_ij = (- 2 * (self.container.dim + 2) * self.viscosity * m_ij
                    / self.container.particle_densities[p_j]
                    / (R.norm_sqr() + 0.01 * self.container.dh**2) 
                    * nabla_ij.outer_product(R) 
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            m_ij = (self.density_0 * self.container.particle_rest_volumes[p_j])
            A_ij = (- 2 * (self.container.dim + 2) * self.viscosity_b * m_ij
                    / self.container.particle_densities[p_i]
                    / (R.norm_sqr() + 0.01 * self.container.dh**2) 
                    * nabla_ij.outer_product(R) 
            )

        return A_ij

    @ti.kernel
    def compute_Ap(self):
        # the linear operator
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_Ap_task, ret)
                ret *= self.dt[None]
                ret /= self.density_0
                ret += self.cg_p[p_i]
                self.cg_Ap[p_i] = ret    
    
    @ti.func
    def compute_Ap_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            A_ij = self.compute_A_ij(p_i, p_j)

            # preconditioner
            ret += self.cg_diagnol_ii_inv[p_i] @ (-A_ij) @ self.cg_p[p_j]

    @ti.kernel
    def compute_cg_alpha(self):
        self.cg_alpha[None] = 0.0
        numerator = 0.0
        denominator = 0.0
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                numerator += self.cg_r[p_i].norm_sqr()
                denominator += self.cg_p[p_i].dot(self.cg_Ap[p_i])

        if denominator > 1e-18:
            self.cg_alpha[None] = numerator / denominator
        else:
            self.cg_alpha[None] = 0.0
    
    @ti.kernel
    def update_cg_x(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_x[p_i] += self.cg_alpha[None] * self.cg_p[p_i]
    
    @ti.kernel
    def update_cg_r_and_beta(self):
        self.cg_error[None] = 0.0
        numerator = 0.0
        denominator = 0.0
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                new_r_i = self.cg_r[p_i] - self.cg_alpha[None] * self.cg_Ap[p_i]
                numerator += new_r_i.norm_sqr()
                denominator += self.cg_r[p_i].norm_sqr()
                self.cg_error[None] += new_r_i.norm_sqr()
                self.cg_r[p_i] = new_r_i
        
        self.cg_error[None] = ti.sqrt(self.cg_error[None])
        if denominator > 1e-18:
            self.cg_beta[None] = numerator / denominator
        else:
            self.cg_beta[None] = 0.0

    @ti.kernel
    def update_p(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_p[p_i] =  self.cg_r[p_i] + self.cg_beta[None] * self.cg_p[p_i]

    @ti.kernel
    def prepare_guess(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_x[p_i] -= self.original_velocity[p_i]

    def conjugate_gradient_loop(self):
        tol = 1000.0
        num_itr = 0

        while tol > self.cg_tol and num_itr < 1000:
            self.compute_Ap()
            self.compute_cg_alpha()
            self.update_cg_x()


            self.update_cg_r_and_beta()
            self.update_p()
            tol = self.cg_error[None]
            num_itr += 1


        print("CG iteration: ", num_itr, " error: ", tol)
            
    @ti.kernel
    def viscosity_update_velocity(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] = self.cg_x[p_i]
                
    @ti.kernel
    def copy_back_original_velocity(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] = self.original_velocity[p_i]

    @ti.kernel
    def add_viscosity_force_to_rigid(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector([0.0 for _ in range(self.container.dim)]) # dummy variable
                self.container.for_all_neighbors(p_i, self.add_viscosity_force_to_rigid_task, ret)
                
    @ti.func
    def add_viscosity_force_to_rigid_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_rigid and self.container.particle_is_dynamic[p_j]:
            pos_i = self.container.particle_positions[p_i]
            pos_j = self.container.particle_positions[p_j]
            # Compute the viscosity force contribution
            R = pos_i - pos_j
            v_ij = self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j]
            nabla_ij = self.kernel_gradient(R)

            acc = (
                2 * (self.container.dim + 2) * self.viscosity_b
                * (self.density_0 * self.container.particle_rest_volumes[p_j])
                / self.container.particle_densities[p_i] /  self.density_0
                * ti.math.dot(v_ij, R)
                / (R.norm_sqr() + 0.01 * self.container.dh**2)
                * nabla_ij
            )

            object_j = self.container.particle_object_ids[p_j]
            center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
            force_j =  - acc * self.container.particle_rest_volumes[p_i] * self.density_0
            torque_j = ti.math.cross(pos_j - center_of_mass_j, force_j)
            self.container.rigid_body_forces[object_j] += force_j
            self.container.rigid_body_torques[object_j] += torque_j


    def implicit_viscosity_solve(self):
        self.prepare_conjugate_gradient_solver1()
        self.compute_Ap()
        self.prepare_conjugate_gradient_solver2()
        self.conjugate_gradient_loop()
        self.viscosity_update_velocity()
        self.compute_viscosity_acceleration_standard() # we use this function to update acceleration
        self.copy_back_original_velocity() # copy back original velocity
        self.prepare_guess()

    ################# End of Implicit Viscosity Solver #################

    @ti.kernel
    def compute_density(self):
        """
        compute density for each particle from mass of neighbors in SPH standard way.
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_densities[p_i] = self.container.particle_rest_volumes[p_i] * self.kernel_W(0.0)
                ret_i = 0.0
                self.container.for_all_neighbors(p_i, self.compute_density_task, ret_i)
                self.container.particle_densities[p_i] += ret_i
                self.container.particle_densities[p_i] *= self.density_0

    @ti.func
    def compute_density_task(self, p_i, p_j, ret: ti.template()):
        # Fluid neighbors and rigid neighbors are treated the same
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        R_mod = R.norm()
        ret += self.container.particle_rest_volumes[p_j] * self.kernel_W(R_mod)

    ################# Enforce Boundary #################
    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.container.particle_velocities[p_i] -= (
            1.0 + c_f) * self.container.particle_velocities[p_i].dot(vec) * vec
        
    @ti.kernel
    def enforce_domain_boundary_2D(self, particle_type:int):
        # make sure the particles are inside the domain
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == particle_type and self.container.particle_is_dynamic[p_i]: 
                pos = self.container.particle_positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.container.domain_size[0] - self.container.padding:
                    collision_normal[0] += 1.0
                    self.container.particle_positions[p_i][0] = self.container.domain_size[0] - self.container.padding
                if pos[0] <= self.container.padding:
                    collision_normal[0] += -1.0
                    self.container.particle_positions[p_i][0] = self.container.padding
                if pos[1] > self.container.domain_size[1] - self.container.padding:
                    collision_normal[1] += 1.0
                    self.container.particle_positions[p_i][1] = self.container.domain_size[1] - self.container.padding
                if pos[1] <= self.container.padding:
                    collision_normal[1] += -1.0
                    self.container.particle_positions[p_i][1] = self.container.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)
    @ti.kernel
    def enforce_domain_boundary_3D(self, particle_type:int):
        # make sure the particles are inside the domain
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == particle_type and self.container.particle_is_dynamic[p_i]:
                pos = self.container.particle_positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.container.domain_size[0] - self.container.padding:
                    collision_normal[0] += 1.0
                    self.container.particle_positions[p_i][0] = self.container.domain_size[0] - self.container.padding
                if pos[0] <= self.container.padding:
                    collision_normal[0] += -1.0
                    self.container.particle_positions[p_i][0] = self.container.padding

                if pos[1] > self.container.domain_size[1] - self.container.padding:
                    collision_normal[1] += 1.0
                    self.container.particle_positions[p_i][1] = self.container.domain_size[1] - self.container.padding
                if pos[1] <= self.container.padding:
                    collision_normal[1] += -1.0
                    self.container.particle_positions[p_i][1] = self.container.padding

                if pos[2] > self.container.domain_size[2] - self.container.padding:
                    collision_normal[2] += 1.0
                    self.container.particle_positions[p_i][2] = self.container.domain_size[2] - self.container.padding
                if pos[2] <= self.container.padding:
                    collision_normal[2] += -1.0
                    self.container.particle_positions[p_i][2] = self.container.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)
    @ti.kernel
    def enforce_domain_boundary_3D(self, particle_type: int):
        ds = self.container.domain_start_ti[None]
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == particle_type and self.container.particle_is_dynamic[p_i]:
                pos = self.container.particle_positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > ds[0] + self.container.domain_size[0] - self.container.padding:
                    collision_normal[0] += 1.0
                    self.container.particle_positions[p_i][0] = self.container.domain_size[0] - self.container.padding
                if pos[0] <= ds[0] + self.container.padding:
                    collision_normal[0] += -1.0
                    self.container.particle_positions[p_i][0] = self.container.padding

                if pos[1] > ds[1] + self.container.domain_size[1] - self.container.padding:
                    collision_normal[1] += 1.0
                    self.container.particle_positions[p_i][1] = self.container.domain_size[1] - self.container.padding
                if pos[1] <= ds[1] + self.container.padding:
                    collision_normal[1] += -1.0
                    self.container.particle_positions[p_i][1] = self.container.padding

                if pos[2] > ds[2] + self.container.domain_size[2] - self.container.padding:
                    collision_normal[2] += 1.0
                    self.container.particle_positions[p_i][2] = self.container.domain_size[2] - self.container.padding
                if pos[2] <= ds[2] + self.container.padding:
                    collision_normal[2] += -1.0
                    self.container.particle_positions[p_i][2] = self.container.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)
                            
        # de = self.container.domain_end_ti[None]
        # pad = self.container.padding_ti[None]

        # L = de - ds
        # eps = 1e-5
        # pad_eff = ti.Vector([0.0, 0.0, 0.0])
        # for k in ti.static(range(3)):
        #     pad_eff[k] = ti.min(pad, 0.49 * L[k])   # ensure low<high

        # low  = ds + pad_eff
        # high = de - pad_eff
        # for k in ti.static(range(3)):
        #     if high[k] - low[k] <= eps:
        #         low[k]  = ds[k] + eps
        #         high[k] = de[k] - eps

        # for p_i in range(self.container.particle_num[None]):
        #     if (self.container.particle_materials[p_i] == particle_type) and (self.container.particle_is_dynamic[p_i] == 1):
        #         pos = self.container.particle_positions[p_i]
        #         nrm = ti.Vector([0.0, 0.0, 0.0])

        #         # X
        #         if pos[0] > high[0]:
        #             nrm[0] += 1.0
        #             self.container.particle_positions[p_i][0] = high[0]
        #         elif pos[0] < low[0]:
        #             nrm[0] += -1.0
        #             self.container.particle_positions[p_i][0] = low[0]

        #         # Y
        #         if pos[1] > high[1]:
        #             nrm[1] += 1.0
        #             self.container.particle_positions[p_i][1] = high[1]
        #         elif pos[1] < low[1]:
        #             nrm[1] += -1.0
        #             self.container.particle_positions[p_i][1] = low[1]

        #         # Z
        #         if pos[2] > high[2]:
        #             nrm[2] += 1.0
        #             self.container.particle_positions[p_i][2] = high[2]
        #         elif pos[2] < low[2]:
        #             nrm[2] += -1.0
        #             self.container.particle_positions[p_i][2] = low[2]

        #         l = ti.sqrt(nrm.dot(nrm))
        #         if l > 1e-6:
        #             self.simulate_collisions(p_i, nrm / l)

    def enforce_domain_boundary(self, particle_type:int):
        if self.container.dim == 2:
            self.enforce_domain_boundary_2D(particle_type)
        elif self.container.dim == 3:
            self.enforce_domain_boundary_3D(particle_type)

    ################# End of Enforce Boundary #################

    @ti.kernel
    def _renew_rigid_particle_state(self):
        # update rigid particle state from rigid body state updated by the rigid solver
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                if self.container.rigid_body_is_dynamic[object_id]:
                    center_of_mass = self.container.rigid_body_centers_of_mass[object_id]
                    rotation = self.container.rigid_body_rotations[object_id]
                    velocity = self.container.rigid_body_velocities[object_id]
                    angular_velocity = self.container.rigid_body_angular_velocities[object_id]
                    q = self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id]
                    p = rotation @ q
                    self.container.particle_positions[p_i] = center_of_mass + p
                    self.container.particle_velocities[p_i] = velocity + ti.math.cross(angular_velocity, p)

    def renew_rigid_particle_state(self):
        self._renew_rigid_particle_state()
        
        if self.cfg.get_cfg("exportObj"):
            for obj_i in range(self.container.object_num[None]):
                if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                    center_of_mass = self.container.rigid_body_centers_of_mass[obj_i]
                    rotation = self.container.rigid_body_rotations[obj_i]
                    ret = rotation.to_numpy() @ (self.container.object_collection[obj_i]["restPosition"] - self.container.object_collection[obj_i]["restCenterOfMass"]).T
                    self.container.object_collection[obj_i]["mesh"].vertices = ret.T + center_of_mass.to_numpy()

    @ti.kernel
    def update_fluid_velocity(self):
        """
        update velocity for each particle from acceleration
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_fluid_position(self):
        """
        update position for each particle from velocity
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]

            elif self.container.particle_positions[p_i][1] > self.g_upper:
                # the emitter part
                obj_id = self.container.particle_object_ids[p_i]
                if self.container.object_materials[obj_id] == self.container.material_fluid:
                    self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
                    if self.container.particle_positions[p_i][1] <= self.g_upper:
                        self.container.particle_materials[p_i] = self.container.material_fluid
        
            
    @ti.kernel
    def prepare_emitter(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                if self.container.particle_positions[p_i][1] > self.g_upper:
                    # an awful hack to realize emitter
                    # not elegant but works
                    # feel free to implement your own emitter
                    self.container.particle_materials[p_i] = self.container.material_rigid

    @ti.kernel
    def init_object_id(self):
        self.container.particle_object_ids.fill(-1)

    def prepare(self):
        self.init_object_id()
        self.container.insert_object()
        self.prepare_emitter()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()
        self.container.prepare_neighborhood_search()
        self.compute_rigid_particle_volume()

    def step(self):
        self._step()
        self.container.total_time += self.dt[None]
        self.rigid_solver.total_time += self.dt[None]
        self.compute_rigid_particle_volume()
