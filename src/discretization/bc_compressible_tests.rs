use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch};
use crate::core::Vector3;
use crate::mesh::{BoundaryMesh, StructuredMesh3d};

#[test]
fn isothermal_wall_ghost_temperature_uses_wall_value() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let viscous = crate::physics::ViscousPhysicsConfig::default();
    let t_owner = 400.0;
    let t_wall = 300.0;
    let spacing = 0.25;
    let t_ghost = crate::discretization::wall_ghost_temperature(
        t_owner,
        WallHeat::Isothermal {
            temperature: t_wall,
        },
        spacing,
        Some(&viscous),
        &eos,
    )
    .expect("t_ghost");
    assert!((t_ghost - t_wall).abs() < 1.0e-10);
}

#[test]
fn wall_no_slip_ghost_velocity_negates_owner() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let params = FreestreamParams {
        mach: 0.2,
        ..FreestreamParams::default()
    };
    let fields = ConservedFields::from_freestream(1, &eos, &params).expect("fields");
    let owner = fields.cell_state(0).expect("cell");
    let owner_prim = crate::field::primitive_from_conserved(&eos, &owner).expect("owner prim");
    let geom = FaceGeometry3d {
        normal: Vector3::new(-1.0, 0.0, 0.0),
        spacing: 0.5,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let p_floor = crate::field::positivity_pressure_floor(params.pressure);
    let fs_ctx = FreestreamContext::new(&eos, None, None);
    let ghost = wall_ghost(
        &owner,
        &geom,
        true,
        WallHeat::Adiabatic,
        &fs_ctx,
        p_floor,
        None,
    )
    .expect("ghost");
    let prim = crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("prim");
    for (g, o) in prim.velocity.iter().zip(owner_prim.velocity.iter()) {
        assert!(
            (g + o).abs() < 1.0e-10,
            "u_g should be -u_o, got {g} vs {o}"
        );
    }
    let u_face = [
        0.5 * (owner_prim.velocity[0] + prim.velocity[0]),
        0.5 * (owner_prim.velocity[1] + prim.velocity[1]),
        0.5 * (owner_prim.velocity[2] + prim.velocity[2]),
    ];
    assert!(u_face.iter().all(|&v| v.abs() < 1.0e-10));
}

#[test]
fn wall_slip_ghost_mirrors_normal_preserves_tangential_at_face() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let p = 101_325.0;
    let t = 300.0;
    let rho = eos.density(p, t).expect("rho");
    let u_owner = [120.0, 45.0, 10.0];
    let prim = PrimitiveState {
        density: rho,
        velocity: u_owner,
        pressure: p,
        temperature: t,
    };
    let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
    let normal = Vector3::new(-1.0, 0.0, 0.0);
    let geom = FaceGeometry3d {
        normal,
        spacing: 0.5,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let fs_ctx = FreestreamContext::new(&eos, None, None);
    let ghost = wall_ghost(
        &owner,
        &geom,
        false,
        WallHeat::Adiabatic,
        &fs_ctx,
        1.0e-6,
        None,
    )
    .expect("ghost");
    let u_g = crate::field::primitive_from_conserved(&eos, &ghost.conserved)
        .expect("ghost prim")
        .velocity;
    let u_f = [
        0.5 * (u_owner[0] + u_g[0]),
        0.5 * (u_owner[1] + u_g[1]),
        0.5 * (u_owner[2] + u_g[2]),
    ];
    let un_face = u_f[0] * normal.x + u_f[1] * normal.y + u_f[2] * normal.z;
    assert!(
        un_face.abs() < 1.0e-10,
        "slip wall face normal velocity should be 0"
    );
    let un_o = u_owner[0] * normal.x + u_owner[1] * normal.y + u_owner[2] * normal.z;
    let u_t_o = [
        u_owner[0] - un_o * normal.x,
        u_owner[1] - un_o * normal.y,
        u_owner[2] - un_o * normal.z,
    ];
    let u_t_f = [
        u_f[0] - un_face * normal.x,
        u_f[1] - un_face * normal.y,
        u_f[2] - un_face * normal.z,
    ];
    for i in 0..3 {
        assert!(
            (u_t_f[i] - u_t_o[i]).abs() < 1.0e-10,
            "tangential at face should match owner, component {i}"
        );
    }
}

#[test]
fn supersonic_inlet_ghost_uses_freestream_static_state() {
    use crate::discretization::freestream_pair::FreestreamPairFixture;

    let pair = FreestreamPairFixture::air_sutherland(8.0);
    let side = pair.inviscid_side();
    let owner = ConservedFields::from_freestream_context(1, &side.ctx, side.fs)
        .expect("fields")
        .cell_state(0)
        .expect("cell");
    let geom = FaceGeometry3d {
        normal: Vector3::new(1.0, 0.0, 0.0),
        spacing: 0.5,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let ghost = inlet_ghost(
        &owner,
        &geom,
        &InletGhostParams {
            supersonic: true,
            velocity_direction: [1.0, 0.0, 0.0],
            freestream: side.fs,
            fs_ctx: &side.ctx,
            total_pressure: 1.0e9,
            total_temperature: 1.0e4,
        },
    )
    .expect("ghost");
    let prim = crate::field::primitive_from_conserved(side.eos, &ghost.conserved).expect("prim");
    let ref_prim = side.ctx.primitive(side.fs).expect("ref");
    assert!((prim.density - ref_prim.density).abs() / ref_prim.density < 1.0e-6);
}

#[test]
fn subsonic_inlet_ghost_ignores_high_mach_freestream() {
    use crate::discretization::freestream_pair::FreestreamPairFixture;

    let pair = FreestreamPairFixture::air_sutherland(0.3);
    let side = pair.inviscid_side();
    let mut high_mach_fs = *side.fs;
    high_mach_fs.mach = 8.0;
    let owner = ConservedFields::from_freestream_context(1, &side.ctx, side.fs)
        .expect("fields")
        .cell_state(0)
        .expect("cell");
    let geom = FaceGeometry3d {
        normal: Vector3::new(1.0, 0.0, 0.0),
        spacing: 0.5,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let owner_prim = side.ctx.primitive(side.fs).expect("owner prim");
    let total_pressure = side
        .eos
        .stagnation_pressure(owner_prim.pressure, side.fs.mach)
        .expect("p0");
    let total_temperature =
        owner_prim.temperature * (1.0 + 0.5 * (side.eos.gamma - 1.0) * side.fs.mach * side.fs.mach);
    let ghost = inlet_ghost(
        &owner,
        &geom,
        &InletGhostParams {
            supersonic: false,
            velocity_direction: [1.0, 0.0, 0.0],
            freestream: &high_mach_fs,
            fs_ctx: &side.ctx,
            total_pressure,
            total_temperature,
        },
    )
    .expect("ghost");
    let prim = crate::field::primitive_from_conserved(side.eos, &ghost.conserved).expect("prim");
    let high_ref = side.ctx.primitive(&high_mach_fs).expect("ref");
    // 亚声速入口走总压/总温分支，不采用 `freestream.mach=8` 的 \(u^*=8\) 来流速度。
    assert!((prim.velocity[0] - high_ref.velocity[0]).abs() > 1.0);
}

#[test]
fn supersonic_outlet_ghost_extrapolates_owner_state() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let prim = eos
        .freestream_primitive(3.0, 25_000.0, 280.0, [1.0, 0.0, 0.0])
        .expect("prim");
    let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
    let geom = FaceGeometry3d {
        normal: Vector3::new(1.0, 0.0, 0.0),
        spacing: 0.5,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let ghost = outlet_ghost(&owner, &geom, 101_325.0, true, &eos, 1.0e-6).expect("ghost");
    let ghost_prim =
        crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("ghost prim");
    assert!((ghost_prim.pressure - prim.pressure).abs() < 1.0e-8);
    assert!((ghost_prim.density - prim.density).abs() < 1.0e-10);
    for i in 0..3 {
        assert!((ghost_prim.velocity[i] - prim.velocity[i]).abs() < 1.0e-10);
    }
}

#[test]
fn subsonic_outlet_ghost_sets_static_pressure() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let prim = eos
        .freestream_primitive(0.3, 90_000.0, 300.0, [1.0, 0.0, 0.0])
        .expect("prim");
    let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
    let geom = FaceGeometry3d {
        normal: Vector3::new(1.0, 0.0, 0.0),
        spacing: 0.5,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let ghost = outlet_ghost(&owner, &geom, 101_325.0, false, &eos, 1.0e-6).expect("ghost");
    let ghost_prim =
        crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("ghost prim");
    assert!((ghost_prim.pressure - 101_325.0).abs() < 1.0e-8);
    assert!(ghost_prim.velocity[0].is_finite());
    assert!((ghost_prim.velocity[1] - prim.velocity[1]).abs() < 1.0e-10);
    assert!((ghost_prim.velocity[2] - prim.velocity[2]).abs() < 1.0e-10);
}

#[test]
fn farfield_supersonic_outflow_uses_owner_state() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let owner_prim = eos
        .freestream_primitive(2.0, 70_000.0, 280.0, [1.0, 0.0, 0.0])
        .expect("owner");
    let owner = ConservedState::from_primitive(&eos, &owner_prim).expect("owner cons");
    let fs = FreestreamParams {
        mach: 0.2,
        pressure: 101_325.0,
        temperature: 288.15,
        ..FreestreamParams::default()
    };
    let geom = FaceGeometry3d {
        normal: Vector3::new(1.0, 0.0, 0.0),
        spacing: 0.5,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let fs_ctx = FreestreamContext::new(&eos, None, None);
    let exterior = farfield_ghost(&owner, &geom, &fs, &fs_ctx).expect("farfield");
    let prim = crate::field::primitive_from_conserved(&eos, &exterior.conserved).expect("prim");
    assert!((prim.pressure - owner_prim.pressure).abs() < 1.0e-8);
    assert!((prim.velocity[0] - owner_prim.velocity[0]).abs() < 1.0e-10);
}

#[test]
fn apply_farfield_patch() {
    use crate::discretization::freestream_pair::FreestreamPairFixture;

    let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
    let pair = FreestreamPairFixture::air_sutherland(0.3);
    let side = pair.inviscid_side();
    let fields = ConservedFields::from_freestream_context(mesh.num_cells(), &side.ctx, side.fs)
        .expect("fields");
    let faces = mesh.resolve_logical_boundary("i_max").expect("faces");
    let first_face = faces[0];
    let patches = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        BoundaryKind::Farfield {
            mach: side.fs.mach,
            pressure: side.fs.pressure,
            temperature: side.fs.temperature,
            alpha: 0.0,
            beta: 0.0,
        },
    )]);
    let mut ghosts = BoundaryGhostBuffer::new();
    apply_compressible_boundary_conditions(
        &mesh,
        &patches,
        &fields,
        &mut ghosts,
        &side.ctx,
        side.fs,
        None,
    )
    .expect("bc");
    assert!(ghosts.get_face(first_face).is_some());
}
