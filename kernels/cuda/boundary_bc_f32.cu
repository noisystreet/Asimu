// 可压缩 BC：device 上由 owner 守恒量写边界面守恒 ghost（消除每步 H2D）。

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct BcPatchParams {
    uint32_t kind;
    uint32_t flags;
    float f0;
    float f1;
    float f2;
    float f3;
    float f4;
    float f5;
    float f6;
    float f7;
};

struct BcFaceStatic {
    uint32_t owner;
    float nx;
    float ny;
    float nz;
    float spacing;
    uint32_t patch_index;
};

struct BoundaryConservedGhost {
    float rho;
    float mx;
    float my;
    float mz;
    float e;
};

struct Prim {
    float rho;
    float p;
    float ux;
    float uy;
    float uz;
    float t;
};

#define BC_KIND_WALL 1u
#define BC_KIND_FARFIELD 2u
#define BC_KIND_INLET 3u
#define BC_KIND_OUTLET 4u
#define BC_KIND_SYMMETRY 5u
#define BC_KIND_COPY_OWNER 6u

#define BC_WALL_HEAT_ADIABATIC 0u
#define BC_WALL_HEAT_ISOTHERMAL 1u
#define BC_WALL_HEAT_FLUX 2u

__device__ inline float dot3(float ax, float ay, float az, float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ inline float static_temperature_f32(float pressure, float rho, float gamma, float gas_r,
                                               float nondim) {
    float r = rho > 1.0e-30f ? rho : 1.0e-30f;
    if (nondim > 0.0f) {
        return pressure / r * gamma;
    }
    return pressure / (r * gas_r);
}

__device__ inline float density_from_pressure_temperature(float pressure, float temperature,
                                                          float gamma) {
    float t = temperature > 1.0e-30f ? temperature : 1.0e-30f;
    return pressure * gamma / t;
}

__device__ inline Prim prim_from_conserved(float gamma, float min_pressure, float gas_r,
                                           float nondim, float rho, float mx, float my, float mz,
                                           float e) {
    Prim out = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (rho <= 0.0f) {
        return out;
    }
    float inv_rho = 1.0f / rho;
    out.ux = mx * inv_rho;
    out.uy = my * inv_rho;
    out.uz = mz * inv_rho;
    float ke = 0.5f * rho * (out.ux * out.ux + out.uy * out.uy + out.uz * out.uz);
    float internal = e - ke;
    if (internal <= 0.0f) {
        return out;
    }
    out.p = (gamma - 1.0f) * internal;
    if (out.p < min_pressure) {
        out.p = min_pressure;
    }
    out.rho = rho;
    out.t = static_temperature_f32(out.p, rho, gamma, gas_r, nondim);
    return out;
}

__device__ inline void cons_from_prim(Prim p, float gamma, BoundaryConservedGhost *out) {
    float ke = 0.5f * p.rho * (p.ux * p.ux + p.uy * p.uy + p.uz * p.uz);
    float internal = p.p / (gamma - 1.0f);
    out->rho = p.rho;
    out->mx = p.rho * p.ux;
    out->my = p.rho * p.uy;
    out->mz = p.rho * p.uz;
    out->e = internal + ke;
}

__device__ inline float sound_speed(Prim p, float gamma) {
    float r = p.rho > 1.0e-30f ? p.rho : 1.0e-30f;
    return sqrtf(gamma * p.p / r);
}

__device__ inline float entropy_constant(Prim p, float gamma) {
    float r = p.rho > 1.0e-30f ? p.rho : 1.0e-30f;
    return p.p / powf(r, gamma);
}

__device__ inline void wall_ghost_velocity(float ux, float uy, float uz, float nx, float ny, float nz,
                                           uint32_t no_slip, float *gx, float *gy, float *gz) {
    float un = dot3(ux, uy, uz, nx, ny, nz);
    float utx = ux - un * nx;
    float uty = uy - un * ny;
    float utz = uz - un * nz;
    float un_g = -un;
    if (no_slip) {
        *gx = -utx + un_g * nx;
        *gy = -uty + un_g * ny;
        *gz = -utz + un_g * nz;
    } else {
        *gx = utx + un_g * nx;
        *gy = uty + un_g * ny;
        *gz = utz + un_g * nz;
    }
}

__device__ inline Prim freestream_primitive(float mach, float pressure, float temperature,
                                            float dir_x, float dir_y, float dir_z) {
    float mag = sqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    if (mag < 1.0e-30f) {
        mag = 1.0f;
    }
    float inv = 1.0f / mag;
    float speed = mach;
    Prim p;
    p.rho = 1.0f;
    p.p = pressure;
    p.t = temperature;
    p.ux = dir_x * inv * speed;
    p.uy = dir_y * inv * speed;
    p.uz = dir_z * inv * speed;
    return p;
}

__device__ inline Prim characteristic_farfield(Prim owner, Prim farfield, float nx, float ny,
                                               float nz, float gamma) {
    float a_o = sound_speed(owner, gamma);
    float a_f = sound_speed(farfield, gamma);
    float un_o = dot3(owner.ux, owner.uy, owner.uz, nx, ny, nz);
    float un_f = dot3(farfield.ux, farfield.uy, farfield.uz, nx, ny, nz);
    if (un_f <= -a_f) {
        return farfield;
    }
    if (un_o >= a_o) {
        return owner;
    }
    float r_plus = un_o + 2.0f * a_o / (gamma - 1.0f);
    float r_minus = un_f - 2.0f * a_f / (gamma - 1.0f);
    float un = 0.5f * (r_plus + r_minus);
    float sound = (0.25f * (gamma - 1.0f) * (r_plus - r_minus));
    if (sound < 1.0e-30f) {
        sound = 1.0e-30f;
    }
    Prim src = un < 0.0f ? farfield : owner;
    float entropy = entropy_constant(src, gamma);
    float un_src = dot3(src.ux, src.uy, src.uz, nx, ny, nz);
    float utx = src.ux - un_src * nx;
    float uty = src.uy - un_src * ny;
    float utz = src.uz - un_src * nz;
    float rho = powf(sound * sound / (gamma * entropy), 1.0f / (gamma - 1.0f));
    float pressure = entropy * powf(rho, gamma);
    Prim out;
    out.rho = rho;
    out.p = pressure;
    out.ux = utx + un * nx;
    out.uy = uty + un * ny;
    out.uz = utz + un * nz;
    out.t = pressure / (rho > 1.0e-30f ? rho : 1.0e-30f);
    return out;
}

__device__ inline Prim characteristic_outlet(Prim owner, float nx, float ny, float nz,
                                             float static_pressure, float gamma) {
    float a_o = sound_speed(owner, gamma);
    float un_o = dot3(owner.ux, owner.uy, owner.uz, nx, ny, nz);
    float outgoing = un_o + 2.0f * a_o / (gamma - 1.0f);
    float entropy = entropy_constant(owner, gamma);
    float rho = powf(static_pressure / entropy, 1.0f / gamma);
    float sound = sqrtf(gamma * static_pressure / (rho > 1.0e-30f ? rho : 1.0e-30f));
    float un = outgoing - 2.0f * sound / (gamma - 1.0f);
    float utx = owner.ux - un_o * nx;
    float uty = owner.uy - un_o * ny;
    float utz = owner.uz - un_o * nz;
    Prim out;
    out.rho = rho;
    out.p = static_pressure;
    out.ux = utx + un * nx;
    out.uy = uty + un * ny;
    out.uz = utz + un * nz;
    out.t = static_pressure / (rho > 1.0e-30f ? rho : 1.0e-30f);
    return out;
}

__device__ inline float inlet_mach_residual(float mach, float outgoing, float normal_projection,
                                            float total_temperature, float gamma, float gas_r) {
    float temp_ratio = 1.0f + 0.5f * (gamma - 1.0f) * mach * mach;
    float sound = sqrtf(gamma * gas_r * total_temperature / temp_ratio);
    return sound * (2.0f / (gamma - 1.0f) + normal_projection * mach) - outgoing;
}

__device__ inline float inlet_mach_from_total(float outgoing, float normal_projection,
                                              float total_temperature, float gamma, float gas_r) {
    float lo = 0.0f;
    float hi = 0.999f;
    float f_lo = inlet_mach_residual(lo, outgoing, normal_projection, total_temperature, gamma,
                                     gas_r);
    float f_hi = inlet_mach_residual(hi, outgoing, normal_projection, total_temperature, gamma,
                                     gas_r);
    if (f_lo * f_hi > 0.0f) {
        return f_lo * f_lo < f_hi * f_hi ? lo : hi;
    }
    for (int k = 0; k < 48; ++k) {
        float mid = 0.5f * (lo + hi);
        float f_mid = inlet_mach_residual(mid, outgoing, normal_projection, total_temperature,
                                          gamma, gas_r);
        if (f_lo * f_mid <= 0.0f) {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    return 0.5f * (lo + hi);
}

__device__ inline Prim subsonic_inlet(Prim owner, float nx, float ny, float nz, float total_pressure,
                                      float total_temperature, float dir_x, float dir_y, float dir_z,
                                      float gamma, float gas_r) {
    float a_o = sound_speed(owner, gamma);
    float un_o = dot3(owner.ux, owner.uy, owner.uz, nx, ny, nz);
    float outgoing = un_o + 2.0f * a_o / (gamma - 1.0f);
    float mag = sqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    if (mag < 1.0e-30f) {
        mag = 1.0f;
    }
    float inv = 1.0f / mag;
    float dx = dir_x * inv;
    float dy = dir_y * inv;
    float dz = dir_z * inv;
    if (dot3(dx, dy, dz, nx, ny, nz) > 0.0f) {
        dx = -dx;
        dy = -dy;
        dz = -dz;
    }
    float normal_projection = dot3(dx, dy, dz, nx, ny, nz);
    float mach = inlet_mach_from_total(outgoing, normal_projection, total_temperature, gamma, gas_r);
    float temp_ratio = 1.0f + 0.5f * (gamma - 1.0f) * mach * mach;
    float static_temperature = total_temperature / temp_ratio;
    if (static_temperature < 1.0e-30f) {
        static_temperature = 1.0e-30f;
    }
    float static_pressure =
        total_pressure * powf(static_temperature / total_temperature, gamma / (gamma - 1.0f));
    float density = static_pressure / (gas_r * static_temperature);
    float speed = mach * sqrtf(gamma * gas_r * static_temperature);
    Prim out;
    out.rho = density;
    out.p = static_pressure;
    out.t = static_temperature;
    out.ux = dx * speed;
    out.uy = dy * speed;
    out.uz = dz * speed;
    return out;
}

__device__ inline Prim apply_wall_bc(Prim owner, BcFaceStatic face, BcPatchParams patch, float gamma,
                                     float nondim) {
    uint32_t no_slip = patch.flags & 1u;
    uint32_t heat_mode = (patch.flags >> 1) & 3u;
    float t_owner = owner.t;
    float t_ghost = t_owner;
    if (heat_mode == BC_WALL_HEAT_ISOTHERMAL) {
        t_ghost = patch.f0;
    } else if (heat_mode == BC_WALL_HEAT_FLUX) {
        float spacing = face.spacing > 0.0f ? face.spacing : 1.0e-30f;
        float lambda = patch.f1 > 0.0f ? patch.f1 : 1.0e-30f;
        t_ghost = t_owner + 2.0f * spacing * patch.f0 / lambda;
    }
    Prim out;
    out.p = owner.p;
    out.t = t_ghost;
    out.rho = density_from_pressure_temperature(owner.p, t_ghost, gamma);
    wall_ghost_velocity(owner.ux, owner.uy, owner.uz, face.nx, face.ny, face.nz, no_slip, &out.ux,
                        &out.uy, &out.uz);
    return out;
}

extern "C" __global__ void apply_compressible_boundary_ghosts_f32(
    uint32_t num_faces, float gamma, float gas_r, float min_pressure, float nondim_flag,
    float fs_mach, float fs_pressure, float fs_temperature, float fs_dir_x, float fs_dir_y,
    float fs_dir_z, const BcFaceStatic *__restrict__ bc_faces,
    const BcPatchParams *__restrict__ bc_patches, const float *__restrict__ cons_rho,
    const float *__restrict__ cons_mx, const float *__restrict__ cons_my,
    const float *__restrict__ cons_mz, const float *__restrict__ cons_e,
    BoundaryConservedGhost *__restrict__ ghost_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    BcFaceStatic face = bc_faces[i];
    BcPatchParams patch = bc_patches[face.patch_index];
    uint32_t owner = face.owner;
    float rho = cons_rho[owner];
    float mx = cons_mx[owner];
    float my = cons_my[owner];
    float mz = cons_mz[owner];
    float e = cons_e[owner];
  if (patch.kind == BC_KIND_COPY_OWNER) {
        ghost_out[i].rho = rho;
        ghost_out[i].mx = mx;
        ghost_out[i].my = my;
        ghost_out[i].mz = mz;
        ghost_out[i].e = e;
        return;
    }
    Prim owner_prim = prim_from_conserved(gamma, min_pressure, gas_r, nondim_flag, rho, mx, my, mz,
                                          e);
    if (owner_prim.rho <= 0.0f) {
        return;
    }
    Prim ghost_prim = owner_prim;
    if (patch.kind == BC_KIND_WALL || patch.kind == BC_KIND_SYMMETRY) {
        BcPatchParams wall_patch = patch;
        if (patch.kind == BC_KIND_SYMMETRY) {
            wall_patch.kind = BC_KIND_WALL;
            wall_patch.flags = 0u;
        }
        ghost_prim = apply_wall_bc(owner_prim, face, wall_patch, gamma, nondim_flag);
    } else if (patch.kind == BC_KIND_OUTLET) {
        if (patch.flags & 1u) {
            ghost_prim = owner_prim;
        } else {
            ghost_prim = characteristic_outlet(owner_prim, face.nx, face.ny, face.nz, patch.f0,
                                               gamma);
        }
    } else if (patch.kind == BC_KIND_FARFIELD) {
        Prim farfield = freestream_primitive(patch.f2, patch.f0, patch.f1, patch.f5, patch.f6,
                                            patch.f7);
        ghost_prim = characteristic_farfield(owner_prim, farfield, face.nx, face.ny, face.nz, gamma);
    } else if (patch.kind == BC_KIND_INLET) {
        if (patch.flags & 1u) {
            Prim farfield = freestream_primitive(fs_mach, fs_pressure, fs_temperature, fs_dir_x,
                                                 fs_dir_y, fs_dir_z);
            ghost_prim = characteristic_farfield(owner_prim, farfield, face.nx, face.ny, face.nz,
                                                 gamma);
        } else {
            ghost_prim =
                subsonic_inlet(owner_prim, face.nx, face.ny, face.nz, patch.f0, patch.f1, patch.f2,
                               patch.f3, patch.f4, gamma, gas_r);
        }
    }
    cons_from_prim(ghost_prim, gamma, &ghost_out[i]);
}
