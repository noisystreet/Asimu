use super::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
use super::gradient::{
    GradientFields, compute_structured_gradients_3d, compute_structured_scalar_gradients_3d,
};
use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::periodic::StructuredPeriodic3d;
use crate::field::PrimitiveFields;
use crate::mesh::StructuredMesh3d;
use crate::physics::IdealGasEoS;

#[test]
fn uniform_flow_has_zero_velocity_gradient() {
    let pair = FreestreamPairFixture::air_sutherland(0.1);
    pair.for_each_inviscid_side(|side| {
        let (mesh, boundary, fields, ghosts) = uniform_farfield_box(4, 4, 4, 1.0, 1.0, 1.0, side);
        let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        prim.fill_from_conserved(&fields, side.eos, side.min_pressure)
            .expect("fill");
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        compute_structured_gradients_3d(
            &mesh,
            &prim,
            side.eos,
            &boundary,
            &ghosts,
            side.min_pressure,
            side.viscous,
            &mut grad,
        )
        .expect("grad");
        for i in 0..mesh.num_cells() {
            let g = grad.velocity_grad_at(i);
            for comp in [g.du, g.dv, g.dw] {
                assert!(
                    comp.iter().all(|&x| x.abs() < 1.0e-10),
                    "{} velocity gradient cell {i}",
                    side.label
                );
            }
        }
    });
}

#[test]
fn linear_field_recovers_constant_structured_gradient() {
    let mesh = StructuredMesh3d::uniform_box("box", 4, 4, 4, 1.0, 1.0, 1.0).expect("mesh");
    let eos = IdealGasEoS::AIR_STANDARD;
    let boundary = BoundarySet::new(Vec::new());
    let ghosts = BoundaryGhostBuffer::new();
    let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                let c = mesh.cell_metric(i, j, k).center;
                prim.density.values_mut()[cell] = 1.0;
                prim.pressure.values_mut()[cell] = 101_325.0;
                prim.velocity_x.values_mut()[cell] = 2.0 * c.x + 3.0 * c.y - 4.0 * c.z;
                prim.velocity_y.values_mut()[cell] = -c.x + 0.5 * c.y + c.z;
                prim.velocity_z.values_mut()[cell] = 7.0 * c.x - 2.0 * c.y + 0.25 * c.z;
            }
        }
    }
    let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
    compute_structured_gradients_3d(
        &mesh, &prim, &eos, &boundary, &ghosts, 1.0e-6, None, &mut grad,
    )
    .expect("grad");
    for cell in 0..mesh.num_cells() {
        let g = grad.velocity_grad_at(cell);
        assert!((g.du[0] - 2.0).abs() < 1.0e-12);
        assert!((g.du[1] - 3.0).abs() < 1.0e-12);
        assert!((g.du[2] + 4.0).abs() < 1.0e-12);
        assert!((g.dv[0] + 1.0).abs() < 1.0e-12);
        assert!((g.dv[1] - 0.5).abs() < 1.0e-12);
        assert!((g.dv[2] - 1.0).abs() < 1.0e-12);
        assert!((g.dw[0] - 7.0).abs() < 1.0e-12);
        assert!((g.dw[1] + 2.0).abs() < 1.0e-12);
        assert!((g.dw[2] - 0.25).abs() < 1.0e-12);
    }
}

#[test]
fn scalar_gradient_recovers_linear_field_on_sheared_mesh() {
    let mesh = sheared_mesh();
    let mut values = vec![0.0; mesh.num_cells()];
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                let c = mesh.cell_metric(i, j, k).center;
                values[cell] = 2.0 * c.x - 3.0 * c.y + 0.5 * c.z;
            }
        }
    }

    let gradients =
        compute_structured_scalar_gradients_3d(&mesh, &values, StructuredPeriodic3d::default());

    for grad in gradients {
        assert!((grad.x - 2.0).abs() < 1.0e-10);
        assert!((grad.y + 3.0).abs() < 1.0e-10);
        assert!((grad.z - 0.5).abs() < 1.0e-10);
    }
}

fn sheared_mesh() -> StructuredMesh3d {
    let nx = 3;
    let ny = 3;
    let nz = 3;
    let mut px = Vec::new();
    let mut py = Vec::new();
    let mut pz = Vec::new();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                px.push(i as Real + 0.25 * j as Real);
                py.push(j as Real + 0.1 * k as Real);
                pz.push(k as Real);
            }
        }
    }
    StructuredMesh3d::new("sheared", nx, ny, nz, px, py, pz).expect("mesh")
}
