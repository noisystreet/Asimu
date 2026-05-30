//! 一维 Euler 方程精确 Riemann 解（Toro 2009 §4）。

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 一维原始变量（x 方向速度）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiemannPrimitive1d {
    pub density: Real,
    pub velocity: Real,
    pub pressure: Real,
}

/// Sod (1978) 经典激波管参数：\(\gamma=1.4\)，\(x=0.5\) 处间断。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SodProblem {
    pub gamma: Real,
    pub left: RiemannPrimitive1d,
    pub right: RiemannPrimitive1d,
}

impl SodProblem {
    pub const CLASSIC: Self = Self {
        gamma: 1.4,
        left: RiemannPrimitive1d {
            density: 1.0,
            velocity: 0.0,
            pressure: 1.0,
        },
        right: RiemannPrimitive1d {
            density: 0.125,
            velocity: 0.0,
            pressure: 0.1,
        },
    };

    #[must_use]
    pub fn riemann_problem(&self) -> RiemannProblem1d {
        RiemannProblem1d {
            gamma: self.gamma,
            left: self.left,
            right: self.right,
        }
    }
}

/// 一般 1D Riemann 问题。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiemannProblem1d {
    pub gamma: Real,
    pub left: RiemannPrimitive1d,
    pub right: RiemannPrimitive1d,
}

struct StarState {
    pressure: Real,
    velocity: Real,
}

struct ShockCoeffs {
    a: Real,
    b: Real,
}

/// 在 \(t>0\) 时于 \(x\) 处采样精确解（\(\xi = x/t\)）。
pub fn sample_exact(problem: &RiemannProblem1d, x: Real, t: Real) -> Result<RiemannPrimitive1d> {
    if t <= 0.0 {
        return Ok(if x < 0.0 { problem.left } else { problem.right });
    }
    let star = solve_star_state(problem)?;
    Ok(sample_at_xi(problem, &star, x / t))
}

/// 经典 Sod 精确解。
pub fn sod_sample(x: Real, t: Real) -> Result<RiemannPrimitive1d> {
    sample_exact(&SodProblem::CLASSIC.riemann_problem(), x, t)
}

/// 求解 1D Riemann 星区压力与速度（供 HLLC 等近似求解器复用）。
pub fn solve_star_pressure_velocity(
    left: RiemannPrimitive1d,
    right: RiemannPrimitive1d,
    gamma: Real,
) -> Result<(Real, Real)> {
    let star = solve_star_state(&RiemannProblem1d { gamma, left, right })?;
    Ok((star.pressure, star.velocity))
}

fn solve_star_state(problem: &RiemannProblem1d) -> Result<StarState> {
    let g = problem.gamma;
    let left = problem.left;
    let right = problem.right;
    validate_state(left)?;
    validate_state(right)?;
    let a_l = sound_speed(left, g);
    let a_r = sound_speed(right, g);
    let mut p = pvrs_initial_pressure(left, right, a_l, a_r, g);
    for _ in 0..64 {
        let f_l = pressure_function(p, left, g);
        let f_r = pressure_function(p, right, g);
        let g_l = pressure_derivative(p, left, g);
        let g_r = pressure_derivative(p, right, g);
        let denom = g_l + g_r;
        if denom.abs() < Real::EPSILON {
            break;
        }
        let dp = -(f_l + f_r + right.velocity - left.velocity) / denom;
        p = (p + dp).max(1.0e-12);
        if dp.abs() < 1.0e-10 * (p + 1.0) {
            break;
        }
    }
    let f_l = pressure_function(p, left, g);
    let f_r = pressure_function(p, right, g);
    let velocity = 0.5 * (left.velocity + right.velocity) + 0.5 * (f_r - f_l);
    Ok(StarState {
        pressure: p,
        velocity,
    })
}

fn validate_state(state: RiemannPrimitive1d) -> Result<()> {
    if state.density <= 0.0 || state.pressure <= 0.0 {
        return Err(AsimuError::Field(
            "Riemann 初值须为正密度与压力".to_string(),
        ));
    }
    Ok(())
}

fn pvrs_initial_pressure(
    left: RiemannPrimitive1d,
    right: RiemannPrimitive1d,
    a_l: Real,
    a_r: Real,
    _gamma: Real,
) -> Real {
    let rho_bar = 0.5 * (left.density + right.density);
    let a_bar = 0.5 * (a_l + a_r);
    let p_pvrs = 0.5 * (left.pressure + right.pressure)
        - 0.125 * (right.velocity - left.velocity) * rho_bar * a_bar;
    let p_min = 1.0e-6;
    let p_max = 2.0 * left.pressure.max(right.pressure);
    p_pvrs.clamp(p_min, p_max.max(p_min))
}

fn sample_at_xi(problem: &RiemannProblem1d, star: &StarState, xi: Real) -> RiemannPrimitive1d {
    if xi <= star.velocity {
        sample_left(problem, star, xi)
    } else {
        sample_right(problem, star, xi)
    }
}

fn sample_left(problem: &RiemannProblem1d, star: &StarState, xi: Real) -> RiemannPrimitive1d {
    let left = problem.left;
    if star.pressure > left.pressure {
        sample_left_shock(left, star, problem.gamma, xi)
    } else {
        sample_left_rarefaction(left, star, problem.gamma, xi)
    }
}

fn sample_right(problem: &RiemannProblem1d, star: &StarState, xi: Real) -> RiemannPrimitive1d {
    let right = problem.right;
    if star.pressure > right.pressure {
        sample_right_shock(right, star, problem.gamma, xi)
    } else {
        sample_right_rarefaction(right, star, problem.gamma, xi)
    }
}

fn sample_left_shock(
    left: RiemannPrimitive1d,
    star: &StarState,
    gamma: Real,
    xi: Real,
) -> RiemannPrimitive1d {
    let a_l = sound_speed(left, gamma);
    let s = left.velocity
        - a_l
            * ((gamma + 1.0) / (2.0 * gamma) * (star.pressure / left.pressure - 1.0) + 1.0).sqrt();
    if xi <= s {
        return left;
    }
    let rho = left.density
        * ((star.pressure / left.pressure + (gamma - 1.0) / (gamma + 1.0))
            / ((gamma - 1.0) / (gamma + 1.0) * star.pressure / left.pressure + 1.0));
    RiemannPrimitive1d {
        density: rho,
        velocity: star.velocity,
        pressure: star.pressure,
    }
}

fn sample_left_rarefaction(
    left: RiemannPrimitive1d,
    star: &StarState,
    gamma: Real,
    xi: Real,
) -> RiemannPrimitive1d {
    let a_l = sound_speed(left, gamma);
    let sh = left.velocity - a_l;
    if xi <= sh {
        return left;
    }
    let rho_star = left.density * (star.pressure / left.pressure).powf(1.0 / gamma);
    let a_star = (gamma * star.pressure / rho_star).sqrt();
    let st = star.velocity - a_star;
    if xi >= st {
        return RiemannPrimitive1d {
            density: rho_star,
            velocity: star.velocity,
            pressure: star.pressure,
        };
    }
    let vel = (2.0 / (gamma + 1.0)) * (a_l + (gamma - 1.0) / 2.0 * left.velocity + xi);
    let a = (2.0 / (gamma + 1.0)) * (a_l + (gamma - 1.0) / 2.0 * (left.velocity - xi));
    let rho = left.density * (a / a_l).powf(2.0 / (gamma - 1.0));
    let pressure = left.pressure * (a / a_l).powf(2.0 * gamma / (gamma - 1.0));
    RiemannPrimitive1d {
        density: rho,
        velocity: vel,
        pressure,
    }
}

fn sample_right_shock(
    right: RiemannPrimitive1d,
    star: &StarState,
    gamma: Real,
    xi: Real,
) -> RiemannPrimitive1d {
    let a_r = sound_speed(right, gamma);
    let s = right.velocity
        + a_r
            * ((gamma + 1.0) / (2.0 * gamma) * (star.pressure / right.pressure - 1.0) + 1.0).sqrt();
    if xi >= s {
        return right;
    }
    let rho = right.density
        * ((star.pressure / right.pressure + (gamma - 1.0) / (gamma + 1.0))
            / ((gamma - 1.0) / (gamma + 1.0) * star.pressure / right.pressure + 1.0));
    RiemannPrimitive1d {
        density: rho,
        velocity: star.velocity,
        pressure: star.pressure,
    }
}

fn sample_right_rarefaction(
    right: RiemannPrimitive1d,
    star: &StarState,
    gamma: Real,
    xi: Real,
) -> RiemannPrimitive1d {
    let a_r = sound_speed(right, gamma);
    let sh = right.velocity + a_r;
    if xi >= sh {
        return right;
    }
    let rho_star = right.density * (star.pressure / right.pressure).powf(1.0 / gamma);
    let a_star = (gamma * star.pressure / rho_star).sqrt();
    let st = star.velocity + a_star;
    if xi <= st {
        return RiemannPrimitive1d {
            density: rho_star,
            velocity: star.velocity,
            pressure: star.pressure,
        };
    }
    let vel = (2.0 / (gamma + 1.0)) * (-a_r + (gamma - 1.0) / 2.0 * right.velocity + xi);
    let a = (2.0 / (gamma + 1.0)) * (a_r - (gamma - 1.0) / 2.0 * (right.velocity - xi));
    let rho = right.density * (a / a_r).powf(2.0 / (gamma - 1.0));
    let pressure = right.pressure * (a / a_r).powf(2.0 * gamma / (gamma - 1.0));
    RiemannPrimitive1d {
        density: rho,
        velocity: vel,
        pressure,
    }
}

fn sound_speed(state: RiemannPrimitive1d, gamma: Real) -> Real {
    (gamma * state.pressure / state.density).sqrt()
}

fn shock_coeffs(state: RiemannPrimitive1d, gamma: Real) -> ShockCoeffs {
    ShockCoeffs {
        a: 2.0 / ((gamma + 1.0) * state.density),
        b: (gamma - 1.0) / (gamma + 1.0) * state.pressure,
    }
}

fn pressure_function(p: Real, state: RiemannPrimitive1d, gamma: Real) -> Real {
    if p > state.pressure {
        let c = shock_coeffs(state, gamma);
        (p - state.pressure) * (c.a / (p + c.b)).sqrt()
    } else {
        let a = sound_speed(state, gamma);
        (2.0 * a / (gamma - 1.0)) * ((p / state.pressure).powf((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    }
}

fn pressure_derivative(p: Real, state: RiemannPrimitive1d, gamma: Real) -> Real {
    if p > state.pressure {
        let c = shock_coeffs(state, gamma);
        let root = (c.a / (p + c.b)).sqrt();
        root * (1.0 - 0.5 * (p - state.pressure) / (p + c.b))
    } else {
        let a = sound_speed(state, gamma);
        (1.0 / (state.density * a)) * (p / state.pressure).powf(-(gamma + 1.0) / (2.0 * gamma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn sod_star_state_matches_reference() {
        let star = solve_star_state(&SodProblem::CLASSIC.riemann_problem()).expect("star");
        assert!(approx_eq(star.pressure, 0.303_13, 1.0e-3));
        assert!(approx_eq(star.velocity, 0.927_45, 1.0e-3));
    }

    #[test]
    fn sod_density_profile_at_t02_is_physical() {
        let left = sod_sample(0.1, 0.2).expect("left");
        let mid = sod_sample(0.25, 0.2).expect("mid");
        let right = sod_sample(0.95, 0.2).expect("right");
        assert!(left.density > mid.density);
        assert!(approx_eq(right.density, 0.125, 1.0e-6));
        assert!(mid.pressure > 0.0);
    }

    #[test]
    fn sod_exact_at_quarter_point_matches_toro_table() {
        let prim = crate::physics::sod_sample(0.25, 0.2).expect("sample");
        assert!((prim.density - 0.266).abs() < 0.02);
        assert!((prim.pressure - 0.303).abs() < 0.02);
    }
}
