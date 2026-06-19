use crate::core::Real;
use crate::error::Result;
use crate::field::{PrimitiveFields, primitive_from_conserved_relaxed};
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

pub(super) fn write_cell_primitive(
    primitives: &mut PrimitiveFields,
    cell: usize,
    state: &ConservedState,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<()> {
    let prim = primitive_from_conserved_relaxed(eos, state, p_floor)?;
    primitives.density.values_mut()[cell] = prim.density;
    primitives.pressure.values_mut()[cell] = prim.pressure;
    primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
    primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
    primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
    Ok(())
}

pub(super) fn restore_cell_primitive(
    primitives: &mut PrimitiveFields,
    cell: usize,
    prim: PrimitiveState,
) {
    primitives.density.values_mut()[cell] = prim.density;
    primitives.pressure.values_mut()[cell] = prim.pressure;
    primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
    primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
    primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
}
