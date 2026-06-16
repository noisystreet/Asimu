// ADR 0017 G1：非结构一阶无粘 Roe / HVL / SLAU2 内面通量 + scatter（f32）。
// 数值语义对齐 src/discretization/roe.rs + inviscid.rs scatter。

#include <cuda_runtime.h>
#include <stdint.h>

struct FaceGeom {
    uint32_t owner;
    uint32_t neighbor;
    float nx;
    float ny;
    float nz;
    float owner_scale;
    float neighbor_scale;
};

struct Prim5 {
    float rho;
    float p;
    float ux;
    float uy;
    float uz;
};

struct Cons5 {
    float rho;
    float mx;
    float my;
    float mz;
    float energy;
};

struct Flux5 {
    float mass;
    float mx;
    float my;
    float mz;
    float energy;
};

__device__ inline float dot3(float ax, float ay, float az, float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ inline void normalize3(float &x, float &y, float &z) {
    float mag = sqrtf(x * x + y * y + z * z);
    if (mag > 1.0e-30f) {
        float inv = 1.0f / mag;
        x *= inv;
        y *= inv;
        z *= inv;
    }
}

__device__ inline void face_tangents(float nx, float ny, float nz, float &t1x, float &t1y, float &t1z,
                                     float &t2x, float &t2y, float &t2z) {
    float rx, ry, rz;
    if (fabsf(nx) < 0.9f) {
        rx = 1.0f;
        ry = 0.0f;
        rz = 0.0f;
    } else {
        rx = 0.0f;
        ry = 1.0f;
        rz = 0.0f;
    }
    t1x = ry * nz - rz * ny;
    t1y = rz * nx - rx * nz;
    t1z = rx * ny - ry * nx;
    normalize3(t1x, t1y, t1z);
    t2x = ny * t1z - nz * t1y;
    t2y = nz * t1x - nx * t1z;
    t2z = nx * t1y - ny * t1x;
    normalize3(t2x, t2y, t2z);
}

__device__ inline Cons5 prim_to_cons(float gamma, const Prim5 &prim) {
    float rho = prim.rho;
    float u2 = prim.ux * prim.ux + prim.uy * prim.uy + prim.uz * prim.uz;
    float e = prim.p / ((gamma - 1.0f) * rho);
    Cons5 c;
    c.rho = rho;
    c.mx = rho * prim.ux;
    c.my = rho * prim.uy;
    c.mz = rho * prim.uz;
    c.energy = rho * e + 0.5f * rho * u2;
    return c;
}

__device__ inline Flux5 physical_flux(const Cons5 &cons, const Prim5 &prim, float nx, float ny, float nz) {
    float un = dot3(prim.ux, prim.uy, prim.uz, nx, ny, nz);
    Flux5 f;
    f.mass = prim.rho * un;
    f.mx = prim.rho * un * prim.ux + prim.p * nx;
    f.my = prim.rho * un * prim.uy + prim.p * ny;
    f.mz = prim.rho * un * prim.uz + prim.p * nz;
    f.energy = (cons.energy + prim.p) * un;
    return f;
}

__device__ inline float harten_fix(float lambda, float delta) {
    float abs_l = fabsf(lambda);
    if (abs_l >= delta) {
        return abs_l;
    }
    return (lambda * lambda + delta * delta) / (2.0f * delta);
}

__device__ inline Flux5 roe_flux_f32(float gamma, bool entropy_fix, float nx, float ny, float nz,
                                     const Prim5 &pl, const Prim5 &pr) {
    Cons5 left = prim_to_cons(gamma, pl);
    Cons5 right = prim_to_cons(gamma, pr);
    Flux5 flux_l = physical_flux(left, pl, nx, ny, nz);
    Flux5 flux_r = physical_flux(right, pr, nx, ny, nz);

    float sqrt_dl = sqrtf(pl.rho);
    float sqrt_dr = sqrtf(pr.rho);
    float inv = 1.0f / (sqrt_dl + sqrt_dr);
    float h_l = (left.energy + pl.p) / pl.rho;
    float h_r = (right.energy + pr.p) / pr.rho;
    float u_roe[3] = {(sqrt_dl * pl.ux + sqrt_dr * pr.ux) * inv,
                      (sqrt_dl * pl.uy + sqrt_dr * pr.uy) * inv,
                      (sqrt_dl * pl.uz + sqrt_dr * pr.uz) * inv};
    float h_roe = (sqrt_dl * h_l + sqrt_dr * h_r) * inv;
    float vel2 = dot3(u_roe[0], u_roe[1], u_roe[2], u_roe[0], u_roe[1], u_roe[2]);
    float gamma_term = fmaxf(h_roe - 0.5f * vel2, 1.0e-6f);
    float a = sqrtf((gamma - 1.0f) * gamma_term);
    float rho_roe = sqrt_dl * sqrt_dr;
    float un_roe = dot3(u_roe[0], u_roe[1], u_roe[2], nx, ny, nz);

    float t1x, t1y, t1z, t2x, t2y, t2z;
    face_tangents(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);

    float un_l = dot3(pl.ux, pl.uy, pl.uz, nx, ny, nz);
    float un_r = dot3(pr.ux, pr.uy, pr.uz, nx, ny, nz);
    float ut1_l = dot3(pl.ux, pl.uy, pl.uz, t1x, t1y, t1z);
    float ut1_r = dot3(pr.ux, pr.uy, pr.uz, t1x, t1y, t1z);
    float ut2_l = dot3(pl.ux, pl.uy, pl.uz, t2x, t2y, t2z);
    float ut2_r = dot3(pr.ux, pr.uy, pr.uz, t2x, t2y, t2z);
    float dp = pr.p - pl.p;
    float drho = pr.rho - pl.rho;
    float dun = un_r - un_l;
    float a2 = a * a;
    float alpha1 = 0.5f * (dp - rho_roe * a * dun) / a2;
    float alpha5 = 0.5f * (dp + rho_roe * a * dun) / a2;
    float alpha2 = drho - dp / a2;
    float alpha3 = rho_roe * (ut1_r - ut1_l);
    float alpha4 = rho_roe * (ut2_r - ut2_l);

    float delta = fmaxf(0.2f * (fabsf(un_roe) + a), 1.0e-30f);
    float l1 = entropy_fix ? harten_fix(un_roe - a, delta) : fabsf(un_roe - a);
    float l5 = entropy_fix ? harten_fix(un_roe + a, delta) : fabsf(un_roe + a);
    float l_mid = fabsf(un_roe);

    float diss_mass = 0.0f;
    float diss_mx = 0.0f;
    float diss_my = 0.0f;
    float diss_mz = 0.0f;
    float diss_e = 0.0f;

    auto add_wave = [&](float l, float alpha, float dm, float dmx, float dmy, float dmz, float de) {
        float s = l * alpha;
        diss_mass += s * dm;
        diss_mx += s * dmx;
        diss_my += s * dmy;
        diss_mz += s * dmz;
        diss_e += s * de;
    };

    // r1 acoustic (-)
    add_wave(l1, alpha1, 1.0f, u_roe[0] - a * nx, u_roe[1] - a * ny, u_roe[2] - a * nz,
             h_roe - a * un_roe);
    // r2 contact
    add_wave(l_mid, alpha2, 1.0f, u_roe[0], u_roe[1], u_roe[2], 0.5f * vel2);
    // r3 shear t1
    float ut1_roe = dot3(u_roe[0], u_roe[1], u_roe[2], t1x, t1y, t1z);
    add_wave(l_mid, alpha3, 0.0f, t1x, t1y, t1z, ut1_roe);
    // r4 shear t2
    float ut2_roe = dot3(u_roe[0], u_roe[1], u_roe[2], t2x, t2y, t2z);
    add_wave(l_mid, alpha4, 0.0f, t2x, t2y, t2z, ut2_roe);
    // r5 acoustic (+)
    add_wave(l5, alpha5, 1.0f, u_roe[0] + a * nx, u_roe[1] + a * ny, u_roe[2] + a * nz,
             h_roe + a * un_roe);

    Flux5 out;
    float half = 0.5f;
    out.mass = half * (flux_l.mass + flux_r.mass) - half * diss_mass;
    out.mx = half * (flux_l.mx + flux_r.mx) - half * diss_mx;
    out.my = half * (flux_l.my + flux_r.my) - half * diss_my;
    out.mz = half * (flux_l.mz + flux_r.mz) - half * diss_mz;
    out.energy = half * (flux_l.energy + flux_r.energy) - half * diss_e;
    return out;
}

__device__ inline void scatter_flux(float *res_rho, float *res_mx, float *res_my, float *res_mz,
                                  float *res_e, uint32_t owner, uint32_t neighbor, float os, float ns,
                                  const Flux5 &f) {
    res_rho[owner] += os * f.mass;
    res_mx[owner] += os * f.mx;
    res_my[owner] += os * f.my;
    res_mz[owner] += os * f.mz;
    res_e[owner] += os * f.energy;
    res_rho[neighbor] += ns * f.mass;
    res_mx[neighbor] += ns * f.mx;
    res_my[neighbor] += ns * f.my;
    res_mz[neighbor] += ns * f.mz;
    res_e[neighbor] += ns * f.energy;
}

struct FaceFrameState {
    float rho;
    float un;
    float ut0;
    float ut1;
    float p;
    float rho_e;
};

struct FaceFrameFlux {
    float mass;
    float normal_momentum;
    float tangential_momentum0;
    float tangential_momentum1;
    float energy;
};

__device__ inline FaceFrameState frame_from_prim(float gamma, const Prim5 &prim, float nx, float ny,
                                                 float nz, float t1x, float t1y, float t1z,
                                                 float t2x, float t2y, float t2z) {
    float rho = prim.rho;
    float un = dot3(prim.ux, prim.uy, prim.uz, nx, ny, nz);
    float ut0 = dot3(prim.ux, prim.uy, prim.uz, t1x, t1y, t1z);
    float ut1 = dot3(prim.ux, prim.uy, prim.uz, t2x, t2y, t2z);
    float u2 = prim.ux * prim.ux + prim.uy * prim.uy + prim.uz * prim.uz;
    float internal = prim.p / (gamma - 1.0f);
    FaceFrameState s;
    s.rho = rho;
    s.un = un;
    s.ut0 = ut0;
    s.ut1 = ut1;
    s.p = prim.p;
    s.rho_e = 0.5f * rho * u2 + internal;
    return s;
}

__device__ inline float sound_speed_frame(float rho, float p, float gamma) {
    return sqrtf(gamma * p / rho);
}

__device__ inline FaceFrameFlux physical_face_frame_flux(const FaceFrameState &s) {
    FaceFrameFlux f;
    f.mass = s.rho * s.un;
    f.normal_momentum = s.rho * s.un * s.un + s.p;
    f.tangential_momentum0 = s.rho * s.un * s.ut0;
    f.tangential_momentum1 = s.rho * s.un * s.ut1;
    f.energy = (s.rho_e + s.p) * s.un;
    return f;
}

__device__ inline float specific_enthalpy_hanel(const FaceFrameState &s, float gamma) {
    float a = sound_speed_frame(s.rho, s.p, gamma);
    return a * a / (gamma - 1.0f) + 0.5f * (s.un * s.un + s.ut0 * s.ut0 + s.ut1 * s.ut1);
}

__device__ inline FaceFrameFlux fvs_positive_hanel(const FaceFrameState &state, float gamma) {
    FaceFrameFlux full = physical_face_frame_flux(state);
    float a = sound_speed_frame(state.rho, state.p, gamma);
    float mach = state.un / a;
    if (mach <= -1.0f) {
        FaceFrameFlux zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        return zero;
    }
    if (mach >= 1.0f) {
        return full;
    }
    float mach_plus = mach + 1.0f;
    float mass_plus = 0.25f * state.rho * a * mach_plus * mach_plus;
    float normal_velocity_plus = ((gamma - 1.0f) * state.un + 2.0f * a) / gamma;
    float h = specific_enthalpy_hanel(state, gamma);
    FaceFrameFlux plus;
    plus.mass = mass_plus;
    plus.normal_momentum = mass_plus * normal_velocity_plus;
    plus.tangential_momentum0 = mass_plus * state.ut0;
    plus.tangential_momentum1 = mass_plus * state.ut1;
    plus.energy = mass_plus * h;
    return plus;
}

__device__ inline FaceFrameFlux fvs_negative_hanel(const FaceFrameState &state, float gamma) {
    FaceFrameFlux full = physical_face_frame_flux(state);
    FaceFrameFlux plus = fvs_positive_hanel(state, gamma);
    FaceFrameFlux minus;
    minus.mass = full.mass - plus.mass;
    minus.normal_momentum = full.normal_momentum - plus.normal_momentum;
    minus.tangential_momentum0 = full.tangential_momentum0 - plus.tangential_momentum0;
    minus.tangential_momentum1 = full.tangential_momentum1 - plus.tangential_momentum1;
    minus.energy = full.energy - plus.energy;
    return minus;
}

__device__ inline FaceFrameFlux add_face_frame_fluxes(const FaceFrameFlux &l, const FaceFrameFlux &r) {
    FaceFrameFlux out;
    out.mass = l.mass + r.mass;
    out.normal_momentum = l.normal_momentum + r.normal_momentum;
    out.tangential_momentum0 = l.tangential_momentum0 + r.tangential_momentum0;
    out.tangential_momentum1 = l.tangential_momentum1 + r.tangential_momentum1;
    out.energy = l.energy + r.energy;
    return out;
}

__device__ inline Flux5 to_global_flux(const FaceFrameFlux &face, float nx, float ny, float nz,
                                       float t1x, float t1y, float t1z, float t2x, float t2y,
                                       float t2z) {
    Flux5 out;
    out.mass = face.mass;
    out.mx = face.normal_momentum * nx + face.tangential_momentum0 * t1x +
             face.tangential_momentum1 * t2x;
    out.my = face.normal_momentum * ny + face.tangential_momentum0 * t1y +
             face.tangential_momentum1 * t2y;
    out.mz = face.normal_momentum * nz + face.tangential_momentum0 * t1z +
             face.tangential_momentum1 * t2z;
    out.energy = face.energy;
    return out;
}

__device__ inline Flux5 hvl_flux_f32(float gamma, float nx, float ny, float nz, const Prim5 &pl,
                                     const Prim5 &pr) {
    float t1x, t1y, t1z, t2x, t2y, t2z;
    face_tangents(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
    FaceFrameState fl = frame_from_prim(gamma, pl, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
    FaceFrameState fr = frame_from_prim(gamma, pr, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
    FaceFrameFlux flp = fvs_positive_hanel(fl, gamma);
    FaceFrameFlux frm = fvs_negative_hanel(fr, gamma);
    FaceFrameFlux face = add_face_frame_fluxes(flp, frm);
    return to_global_flux(face, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
}

__device__ inline float speed_magnitude_slau2(const FaceFrameState &s) {
    return sqrtf(s.un * s.un + s.ut0 * s.ut0 + s.ut1 * s.ut1);
}

__device__ inline float specific_enthalpy_slau2(const FaceFrameState &s, float gamma) {
    float speed_sq = s.un * s.un + s.ut0 * s.ut0 + s.ut1 * s.ut1;
    return gamma / (gamma - 1.0f) * s.p / s.rho + 0.5f * speed_sq;
}

__device__ inline float supersonic_alpha_slau2(float mach) {
    return (fabsf(mach) >= 1.0f) ? 0.0f : 1.0f;
}

__device__ inline float signum_slau2(float x) {
    return (x > 0.0f) - (x < 0.0f);
}

__device__ inline float pressure_beta_plus_slau2(float mach, float alpha) {
    float s = signum_slau2(mach);
    float mp1 = mach + 1.0f;
    return (1.0f - alpha) * 0.5f * (1.0f + s) + alpha * 0.25f * (2.0f - mach) * mp1 * mp1;
}

__device__ inline float pressure_beta_minus_slau2(float mach, float alpha) {
    return pressure_beta_plus_slau2(-mach, alpha);
}

__device__ inline float mass_coupling_g_slau2(float ml, float mr) {
    float left = fmaxf(fminf(ml, 0.0f), -1.0f);
    float right = fminf(fmaxf(mr, 0.0f), 1.0f);
    return -left * right;
}

__device__ inline float mass_pressure_xi_slau2(float speed_l, float speed_r, float c) {
    float speed = sqrtf(0.5f * (speed_l * speed_l + speed_r * speed_r));
    float m_cap = fminf(speed / c, 1.0f);
    float one_minus = 1.0f - m_cap;
    return one_minus * one_minus;
}

__device__ inline float slau2_pressure_dissipation(float speed_l, float speed_r, float c) {
    float speed = sqrtf(0.5f * (speed_l * speed_l + speed_r * speed_r));
    return fminf(speed / c, 1.0f);
}

__device__ inline float interface_pressure_slau2(const FaceFrameState &left,
                                                 const FaceFrameState &right, float c) {
    float ml = left.un / c;
    float mr = right.un / c;
    float alpha_l = supersonic_alpha_slau2(ml);
    float alpha_r = supersonic_alpha_slau2(mr);
    float p_plus_l = pressure_beta_plus_slau2(ml, alpha_l);
    float p_minus_r = pressure_beta_minus_slau2(mr, alpha_r);
    float dp = right.p - left.p;
    float p_bar = 0.5f * (left.p + right.p);
    float diss = slau2_pressure_dissipation(speed_magnitude_slau2(left), speed_magnitude_slau2(right),
                                            c) *
                 (p_plus_l + p_minus_r - 1.0f) * p_bar;
    return p_bar - 0.5f * (p_plus_l - p_minus_r) * dp + diss;
}

__device__ inline FaceFrameFlux slau2_face_flux(const FaceFrameState &left,
                                                const FaceFrameState &right, float gamma) {
    float c_l = fmaxf(sqrtf(gamma * left.p / left.rho), 1.0e-12f);
    float c_r = fmaxf(sqrtf(gamma * right.p / right.rho), 1.0e-12f);
    float c = 0.5f * (c_l + c_r);
    float ml = left.un / c;
    float mr = right.un / c;
    float dp = right.p - left.p;
    float g = mass_coupling_g_slau2(ml, mr);
    float vn_abs =
        (left.rho * fabsf(left.un) + right.rho * fabsf(right.un)) / (left.rho + right.rho);
    float vn_abs_l = (1.0f - g) * vn_abs + g * fabsf(left.un);
    float vn_abs_r = (1.0f - g) * vn_abs + g * fabsf(right.un);
    float xi = mass_pressure_xi_slau2(speed_magnitude_slau2(left), speed_magnitude_slau2(right), c);
    float mass =
        0.5f * (left.rho * (left.un + vn_abs_l) + right.rho * (right.un - vn_abs_r) - xi * dp / c);
    float p_face = interface_pressure_slau2(left, right, c);
    float hl = specific_enthalpy_slau2(left, gamma);
    float hr = specific_enthalpy_slau2(right, gamma);
    float mass_plus = 0.5f * (mass + fabsf(mass));
    float mass_minus = 0.5f * (mass - fabsf(mass));
    FaceFrameFlux f;
    f.mass = mass;
    f.normal_momentum = mass_plus * left.un + mass_minus * right.un + p_face;
    f.tangential_momentum0 = mass_plus * left.ut0 + mass_minus * right.ut0;
    f.tangential_momentum1 = mass_plus * left.ut1 + mass_minus * right.ut1;
    f.energy = mass_plus * hl + mass_minus * hr;
    return f;
}

__device__ inline Flux5 slau2_flux_f32(float gamma, float nx, float ny, float nz, const Prim5 &pl,
                                      const Prim5 &pr) {
    float t1x, t1y, t1z, t2x, t2y, t2z;
    face_tangents(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
    FaceFrameState fl = frame_from_prim(gamma, pl, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
    FaceFrameState fr = frame_from_prim(gamma, pr, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
    FaceFrameFlux face = slau2_face_flux(fl, fr, gamma);
    return to_global_flux(face, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z);
}

// flux_scheme: 0=Roe, 1=Hanel-Van Leer, 2=SLAU2
extern "C" __global__ void inviscid_first_order_bucket_f32(
    const uint32_t *__restrict__ bucket_faces, uint32_t num_faces,
    const FaceGeom *__restrict__ face_geom, const float *__restrict__ prim_rho,
    const float *__restrict__ prim_p, const float *__restrict__ prim_ux,
    const float *__restrict__ prim_uy, const float *__restrict__ prim_uz, float *__restrict__ res_rho,
    float *__restrict__ res_mx, float *__restrict__ res_my, float *__restrict__ res_mz,
    float *__restrict__ res_e, float gamma, uint32_t flux_scheme, uint32_t entropy_fix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    uint32_t face_idx = bucket_faces[i];
    FaceGeom g = face_geom[face_idx];
    if (g.owner_scale == 0.0f && g.neighbor_scale == 0.0f) {
        return;
    }
    float nx = g.nx;
    float ny = g.ny;
    float nz = g.nz;
    normalize3(nx, ny, nz);

    Prim5 pl = {prim_rho[g.owner], prim_p[g.owner], prim_ux[g.owner], prim_uy[g.owner],
                prim_uz[g.owner]};
    Prim5 pr = {prim_rho[g.neighbor], prim_p[g.neighbor], prim_ux[g.neighbor], prim_uy[g.neighbor],
                prim_uz[g.neighbor]};

    Flux5 flux;
    if (flux_scheme == 0u) {
        flux = roe_flux_f32(gamma, entropy_fix != 0u, nx, ny, nz, pl, pr);
    } else if (flux_scheme == 1u) {
        flux = hvl_flux_f32(gamma, nx, ny, nz, pl, pr);
    } else {
        flux = slau2_flux_f32(gamma, nx, ny, nz, pl, pr);
    }
    scatter_flux(res_rho, res_mx, res_my, res_mz, res_e, g.owner, g.neighbor, g.owner_scale,
                 g.neighbor_scale, flux);
}

struct BoundaryFaceGeom {
    uint32_t owner;
    float nx;
    float ny;
    float nz;
    float owner_scale;
};

struct GhostPrim5 {
    float rho;
    float pressure;
    float u;
    float v;
    float w;
};

__device__ inline void scatter_flux_boundary_atomic(float *res_rho, float *res_mx, float *res_my,
                                                    float *res_mz, float *res_e, uint32_t owner,
                                                    float os, const Flux5 &f) {
    atomicAdd(&res_rho[owner], os * f.mass);
    atomicAdd(&res_mx[owner], os * f.mx);
    atomicAdd(&res_my[owner], os * f.my);
    atomicAdd(&res_mz[owner], os * f.mz);
    atomicAdd(&res_e[owner], os * f.energy);
}

// 边界面一阶无粘通量（ghost 每步 H2D；owner 侧 atomic scatter）。
extern "C" __global__ void inviscid_first_order_boundary_f32(
    const BoundaryFaceGeom *__restrict__ faces, uint32_t num_faces,
    const GhostPrim5 *__restrict__ ghosts, const float *__restrict__ prim_rho,
    const float *__restrict__ prim_p, const float *__restrict__ prim_ux,
    const float *__restrict__ prim_uy, const float *__restrict__ prim_uz, float *__restrict__ res_rho,
    float *__restrict__ res_mx, float *__restrict__ res_my, float *__restrict__ res_mz,
    float *__restrict__ res_e, float gamma, uint32_t flux_scheme, uint32_t entropy_fix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    BoundaryFaceGeom g = faces[i];
    if (g.owner_scale == 0.0f) {
        return;
    }
    float nx = g.nx;
    float ny = g.ny;
    float nz = g.nz;
    normalize3(nx, ny, nz);
    uint32_t o = g.owner;
    Prim5 pl = {prim_rho[o], prim_p[o], prim_ux[o], prim_uy[o], prim_uz[o]};
    GhostPrim5 gh = ghosts[i];
    Prim5 pr = {gh.rho, gh.pressure, gh.u, gh.v, gh.w};
    Flux5 flux;
    if (flux_scheme == 0u) {
        flux = roe_flux_f32(gamma, entropy_fix != 0u, nx, ny, nz, pl, pr);
    } else if (flux_scheme == 1u) {
        flux = hvl_flux_f32(gamma, nx, ny, nz, pl, pr);
    } else {
        flux = slau2_flux_f32(gamma, nx, ny, nz, pl, pr);
    }
    scatter_flux_boundary_atomic(res_rho, res_mx, res_my, res_mz, res_e, o, g.owner_scale, flux);
}
