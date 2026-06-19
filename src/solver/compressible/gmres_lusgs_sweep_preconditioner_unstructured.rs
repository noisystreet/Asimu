//! 非结构 GMRES LU-SGS 双扫左预条件器（冻结谱半径系数，线性近似）。

use crate::core::Real;
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::linalg::Preconditioner;
use crate::linalg::ensure_vector_len;
use crate::physics::{ConservedState, IdealGasEoS};
use crate::solver::LuSgsUnstructuredCouplings;
use crate::solver::compressible::gmres_implicit_3d::CONSERVED_COMPONENTS_3D;
use crate::solver::compressible::gmres_implicit_3d::gmres_implicit_typed_common::{
    assign_vector_to_residual, fields_delta_to_vector,
};
use crate::solver::compressible::lu_sgs_sweep_unstructured::{
    LuSgsSweepGmresPreconditionerParams, LuSgsSweepUnstructuredInput,
    LuSgsUnstructuredCouplingsRef, lu_sgs_sweep_unstructured_gmres_preconditioner,
};

pub(crate) struct LusgsSweepUnstructuredGmresPreconditionerBuild {
    pub eos: IdealGasEoS,
    pub couplings: LuSgsUnstructuredCouplings,
    pub base: ConservedFields,
    pub frozen_primitives: PrimitiveFields,
    pub dt: Vec<Real>,
    pub sigma: Vec<Real>,
    pub volumes: Vec<Real>,
    pub solver_order: Vec<usize>,
    pub solver_rank: Vec<usize>,
    pub omega: Real,
    pub backward_damping: Real,
    pub inv_dt_phys: Real,
}

pub(crate) struct LusgsSweepUnstructuredGmresPreconditioner {
    eos: IdealGasEoS,
    couplings: LuSgsUnstructuredCouplings,
    base: ConservedFields,
    sweep: ConservedFields,
    rhs: ConservedResidual,
    frozen_primitives: PrimitiveFields,
    dt: Vec<Real>,
    sigma: Vec<Real>,
    volumes: Vec<Real>,
    solver_order: Vec<usize>,
    solver_rank: Vec<usize>,
    omega: Real,
    backward_damping: Real,
    inv_dt_phys: Real,
}

impl LusgsSweepUnstructuredGmresPreconditioner {
    pub(crate) fn new(params: LusgsSweepUnstructuredGmresPreconditionerBuild) -> Result<Self> {
        let n = params.base.num_cells();
        Ok(Self {
            eos: params.eos,
            couplings: params.couplings,
            base: params.base,
            sweep: ConservedFields::uniform(
                n,
                ConservedState {
                    density: 0.0,
                    momentum: [0.0; 3],
                    total_energy: 0.0,
                },
            )?,
            rhs: ConservedResidual::zeros(n)?,
            frozen_primitives: params.frozen_primitives,
            dt: params.dt,
            sigma: params.sigma,
            volumes: params.volumes,
            solver_order: params.solver_order,
            solver_rank: params.solver_rank,
            omega: params.omega,
            backward_damping: params.backward_damping,
            inv_dt_phys: params.inv_dt_phys,
        })
    }
}

impl Preconditioner for LusgsSweepUnstructuredGmresPreconditioner {
    fn dimension(&self) -> usize {
        self.base.num_cells() * CONSERVED_COMPONENTS_3D
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.dimension(), "gmres lusgs sweep rhs")?;
        ensure_vector_len(out, self.dimension(), "gmres lusgs sweep out")?;
        assign_vector_to_residual(&mut self.rhs, rhs)?;
        lu_sgs_sweep_unstructured_gmres_preconditioner(
            &mut self.sweep,
            &self.base,
            &self.rhs,
            &LuSgsSweepGmresPreconditionerParams {
                eos: &self.eos,
                frozen_primitives: &self.frozen_primitives,
                backward_damping: self.backward_damping,
            },
            LuSgsSweepUnstructuredInput {
                dt: &self.dt,
                sigma: &self.sigma,
                volumes: &self.volumes,
                couplings: LuSgsUnstructuredCouplingsRef::F64(&self.couplings),
                solver_order: &self.solver_order,
                solver_rank: &self.solver_rank,
                omega: self.omega,
                gamma: self.eos.gamma,
                inv_dt_phys: self.inv_dt_phys,
            },
        )?;
        fields_delta_to_vector(&self.base, &self.sweep, out)
    }
}
