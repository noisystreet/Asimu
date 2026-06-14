//! 不可压缩 benchmark 中心线剖面与误差诊断。

use crate::core::Real;
use crate::field::IncompressibleFields;
use crate::mesh::StructuredMesh3d;

use super::benchmark::KnownIncompressibleBenchmark;
use super::incompressible_3d::{
    IncompressibleCenterlineProfileError, IncompressibleCenterlineProfiles,
    IncompressibleLineSample, IncompressibleProfileError,
};

pub(crate) fn incompressible_centerline_profiles(
    benchmark: Option<KnownIncompressibleBenchmark>,
    mesh: &StructuredMesh3d,
    kinematic_viscosity: Real,
    body_force: [Real; 3],
    fields: &IncompressibleFields,
) -> Option<IncompressibleCenterlineProfiles> {
    let benchmark = benchmark.filter(|benchmark| benchmark.emits_centerline_profiles());
    match benchmark {
        Some(KnownIncompressibleBenchmark::LidDrivenCavityRe100) => {
            Some(lid_cavity_centerline_profiles(mesh, fields))
        }
        Some(KnownIncompressibleBenchmark::ChannelPoiseuille)
            if kinematic_viscosity > 0.0 && body_force[0].abs() > 0.0 =>
        {
            Some(channel_poiseuille_centerline_profiles(mesh, fields))
        }
        _ => None,
    }
}

fn lid_cavity_centerline_profiles(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
) -> IncompressibleCenterlineProfiles {
    let k_mid = mesh.nz / 2;
    let target_x = 0.5 * (mesh.node_x(0, 0, k_mid) + mesh.node_x(mesh.nx, 0, k_mid));
    let target_y = 0.5 * (mesh.node_y(0, 0, k_mid) + mesh.node_y(0, mesh.ny, k_mid));
    let mut vertical_u = Vec::with_capacity(mesh.ny);
    for j in 0..mesh.ny {
        let sample = sample_row_at_x(mesh, fields, j, k_mid, target_x);
        vertical_u.push(IncompressibleLineSample {
            coordinate: sample.coordinate,
            velocity_x: sample.velocity[0],
            velocity_y: sample.velocity[1],
            velocity_z: sample.velocity[2],
        });
    }
    let mut horizontal_v = Vec::with_capacity(mesh.nx);
    for i in 0..mesh.nx {
        let sample = sample_column_at_y(mesh, fields, i, k_mid, target_y);
        horizontal_v.push(IncompressibleLineSample {
            coordinate: sample.coordinate,
            velocity_x: sample.velocity[0],
            velocity_y: sample.velocity[1],
            velocity_z: sample.velocity[2],
        });
    }
    IncompressibleCenterlineProfiles {
        vertical_u,
        horizontal_v,
    }
}

#[derive(Debug, Clone, Copy)]
struct InterpolatedLineSample {
    coordinate: Real,
    velocity: [Real; 3],
}

fn sample_row_at_x(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    j: usize,
    k: usize,
    target_x: Real,
) -> InterpolatedLineSample {
    let mut best_left = 0usize;
    for i in 0..mesh.nx.saturating_sub(1) {
        let x0 = mesh.cell_metric(i, j, k).center.x;
        let x1 = mesh.cell_metric(i + 1, j, k).center.x;
        if (target_x >= x0 && target_x <= x1) || (target_x >= x1 && target_x <= x0) {
            best_left = i;
            break;
        }
        if (x0 - target_x).abs() < (mesh.cell_metric(best_left, j, k).center.x - target_x).abs() {
            best_left = i;
        }
    }
    interpolate_between_cells(
        mesh,
        fields,
        InterpolationRequest {
            a: (best_left, j, k),
            b: ((best_left + 1).min(mesh.nx - 1), j, k),
            target: target_x,
            axis: InterpolationAxis::X,
        },
    )
}

fn sample_column_at_y(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    i: usize,
    k: usize,
    target_y: Real,
) -> InterpolatedLineSample {
    let mut best_lower = 0usize;
    for j in 0..mesh.ny.saturating_sub(1) {
        let y0 = mesh.cell_metric(i, j, k).center.y;
        let y1 = mesh.cell_metric(i, j + 1, k).center.y;
        if (target_y >= y0 && target_y <= y1) || (target_y >= y1 && target_y <= y0) {
            best_lower = j;
            break;
        }
        if (y0 - target_y).abs() < (mesh.cell_metric(i, best_lower, k).center.y - target_y).abs() {
            best_lower = j;
        }
    }
    interpolate_between_cells(
        mesh,
        fields,
        InterpolationRequest {
            a: (i, best_lower, k),
            b: (i, (best_lower + 1).min(mesh.ny - 1), k),
            target: target_y,
            axis: InterpolationAxis::Y,
        },
    )
}

#[derive(Debug, Clone, Copy)]
enum InterpolationAxis {
    X,
    Y,
}

#[derive(Debug, Clone, Copy)]
struct InterpolationRequest {
    a: (usize, usize, usize),
    b: (usize, usize, usize),
    target: Real,
    axis: InterpolationAxis,
}

fn interpolate_between_cells(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    request: InterpolationRequest,
) -> InterpolatedLineSample {
    let center_a = mesh
        .cell_metric(request.a.0, request.a.1, request.a.2)
        .center;
    let center_b = mesh
        .cell_metric(request.b.0, request.b.1, request.b.2)
        .center;
    let coord_a = match request.axis {
        InterpolationAxis::X => center_a.x,
        InterpolationAxis::Y => center_a.y,
    };
    let coord_b = match request.axis {
        InterpolationAxis::X => center_b.x,
        InterpolationAxis::Y => center_b.y,
    };
    let t = if (coord_b - coord_a).abs() <= Real::EPSILON {
        0.0
    } else {
        ((request.target - coord_a) / (coord_b - coord_a)).clamp(0.0, 1.0)
    };
    let cell_a = mesh.cell_index(request.a.0, request.a.1, request.a.2);
    let cell_b = mesh.cell_index(request.b.0, request.b.1, request.b.2);
    InterpolatedLineSample {
        coordinate: match request.axis {
            InterpolationAxis::X => center_a.y + t * (center_b.y - center_a.y),
            InterpolationAxis::Y => center_a.x + t * (center_b.x - center_a.x),
        },
        velocity: [
            lerp(
                fields.velocity_x.values()[cell_a],
                fields.velocity_x.values()[cell_b],
                t,
            ),
            lerp(
                fields.velocity_y.values()[cell_a],
                fields.velocity_y.values()[cell_b],
                t,
            ),
            lerp(
                fields.velocity_z.values()[cell_a],
                fields.velocity_z.values()[cell_b],
                t,
            ),
        ],
    }
}

fn lerp(a: Real, b: Real, t: Real) -> Real {
    a + t * (b - a)
}

fn channel_poiseuille_centerline_profiles(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
) -> IncompressibleCenterlineProfiles {
    let i_mid = mesh.nx / 2;
    let k_mid = mesh.nz / 2;
    let mut vertical_u = Vec::with_capacity(mesh.ny);
    for j in 0..mesh.ny {
        let cell = mesh.cell_index(i_mid, j, k_mid);
        vertical_u.push(IncompressibleLineSample {
            coordinate: cell_center_y(mesh, i_mid, j, k_mid),
            velocity_x: fields.velocity_x.values()[cell],
            velocity_y: fields.velocity_y.values()[cell],
            velocity_z: fields.velocity_z.values()[cell],
        });
    }
    IncompressibleCenterlineProfiles {
        vertical_u,
        horizontal_v: Vec::new(),
    }
}

pub(crate) fn poiseuille_profile_error(
    benchmark: Option<KnownIncompressibleBenchmark>,
    mesh: &StructuredMesh3d,
    kinematic_viscosity: Real,
    body_force: [Real; 3],
    fields: &IncompressibleFields,
) -> Option<IncompressibleProfileError> {
    if benchmark != Some(KnownIncompressibleBenchmark::ChannelPoiseuille)
        || kinematic_viscosity <= 0.0
        || body_force[0].abs() <= Real::EPSILON
    {
        return None;
    }
    let y_min = mesh.node_y(0, 0, 0);
    let y_max = mesh.node_y(0, mesh.ny, 0);
    let height = y_max - y_min;
    if height <= 0.0 || mesh.ny <= 2 {
        return None;
    }
    let i_mid = mesh.nx / 2;
    let k_mid = mesh.nz / 2;
    let mut max_abs: Real = 0.0;
    let mut sum_sq: Real = 0.0;
    for j in 1..(mesh.ny - 1) {
        let y = cell_center_y(mesh, i_mid, j, k_mid) - y_min;
        let expected = body_force[0] * y * (height - y) / (2.0 * kinematic_viscosity);
        let cell = mesh.cell_index(i_mid, j, k_mid);
        let error = fields.velocity_x.values()[cell] - expected;
        max_abs = max_abs.max(error.abs());
        sum_sq += error * error;
    }
    Some(IncompressibleProfileError {
        max_abs,
        l2: (sum_sq / (mesh.ny - 2) as Real).sqrt(),
    })
}

pub(crate) fn lid_cavity_profile_error(
    benchmark: Option<KnownIncompressibleBenchmark>,
    profiles: Option<&IncompressibleCenterlineProfiles>,
) -> Option<IncompressibleCenterlineProfileError> {
    if benchmark != Some(KnownIncompressibleBenchmark::LidDrivenCavityRe100) {
        return None;
    }
    let profiles = profiles?;
    Some(IncompressibleCenterlineProfileError {
        vertical_u: profile_error_against_reference(
            &profiles.vertical_u,
            &GHIA_RE100_VERTICAL_U,
            |sample| sample.velocity_x,
        )?,
        horizontal_v: profile_error_against_reference(
            &profiles.horizontal_v,
            &GHIA_RE100_HORIZONTAL_V,
            |sample| sample.velocity_y,
        )?,
    })
}

fn profile_error_against_reference(
    samples: &[IncompressibleLineSample],
    reference: &[(Real, Real)],
    value: impl Fn(&IncompressibleLineSample) -> Real,
) -> Option<IncompressibleProfileError> {
    if samples.is_empty() || reference.len() < 2 {
        return None;
    }
    let mut max_abs: Real = 0.0;
    let mut sum_sq: Real = 0.0;
    let mut count = 0usize;
    for &(coordinate, expected) in reference {
        let Some(actual) = interpolate_profile_sample(samples, coordinate, &value) else {
            continue;
        };
        let error = actual - expected;
        max_abs = max_abs.max(error.abs());
        sum_sq += error * error;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some(IncompressibleProfileError {
        max_abs,
        l2: (sum_sq / count as Real).sqrt(),
    })
}

fn interpolate_profile_sample(
    samples: &[IncompressibleLineSample],
    coordinate: Real,
    value: &impl Fn(&IncompressibleLineSample) -> Real,
) -> Option<Real> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.coordinate.total_cmp(&b.coordinate));
    for pair in sorted.windows(2) {
        let x0 = pair[0].coordinate;
        let x1 = pair[1].coordinate;
        if coordinate >= x0 && coordinate <= x1 {
            let t = if (x1 - x0).abs() <= Real::EPSILON {
                0.0
            } else {
                (coordinate - x0) / (x1 - x0)
            };
            return Some(lerp(value(&pair[0]), value(&pair[1]), t));
        }
    }
    None
}

const GHIA_RE100_VERTICAL_U: [(Real, Real); 17] = [
    (1.0, 1.0),
    (0.9766, 0.84123),
    (0.9688, 0.78871),
    (0.9609, 0.73722),
    (0.9531, 0.68717),
    (0.8516, 0.23151),
    (0.7344, 0.00332),
    (0.6172, -0.13641),
    (0.5, -0.20581),
    (0.4531, -0.2109),
    (0.2813, -0.15662),
    (0.1719, -0.1015),
    (0.1016, -0.06434),
    (0.0703, -0.04775),
    (0.0625, -0.04192),
    (0.0547, -0.03717),
    (0.0, 0.0),
];

const GHIA_RE100_HORIZONTAL_V: [(Real, Real); 17] = [
    (1.0, 0.0),
    (0.9688, -0.05906),
    (0.9609, -0.07391),
    (0.9531, -0.08864),
    (0.9453, -0.10313),
    (0.9063, -0.16914),
    (0.8594, -0.22445),
    (0.8047, -0.24533),
    (0.5, 0.05454),
    (0.2344, 0.17527),
    (0.2266, 0.17507),
    (0.1563, 0.16077),
    (0.0938, 0.12317),
    (0.0781, 0.1089),
    (0.0703, 0.10091),
    (0.0625, 0.09233),
    (0.0, 0.0),
];

fn cell_center_y(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> Real {
    0.5 * (mesh.node_y(i, j, k) + mesh.node_y(i, j + 1, k))
}
