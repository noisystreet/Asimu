//! CUDA 粘性 scratch 扩展（P0：与主文件拆分以满足复杂度门禁）。

use crate::core::Real;
use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;

use super::ViscousAssemblyUnstructuredScratch;

impl ViscousAssemblyUnstructuredScratch {
    pub(crate) fn init_cuda_viscous_topo_from_mesh_cache(
        &mut self,
        mesh_cache: &UnstructuredSolverMeshCache,
    ) {
        if self.cuda_viscous_topo.is_none() {
            self.cuda_viscous_topo = Some(mesh_cache.cuda_viscous_interior_topo.clone());
        }
    }

    pub(crate) fn apply_transport_to_cuda_viscous_topo(&mut self, constant: Option<(Real, Real)>) {
        if let Some((m, l)) = constant {
            let mu = m as f32;
            let lambda = l as f32;
            for face in self
                .cuda_viscous_topo
                .as_mut()
                .expect("cuda viscous topo initialized")
                .faces
                .iter_mut()
            {
                face.mu = mu;
                face.lambda = lambda;
            }
        } else {
            let topo = self
                .cuda_viscous_topo
                .as_mut()
                .expect("cuda viscous topo initialized");
            for (face_idx, face) in topo.faces.iter_mut().enumerate() {
                face.mu = self.face_mu[face_idx] as f32;
                face.lambda = self.face_lambda[face_idx] as f32;
            }
        }
    }

    pub(crate) fn cuda_viscous_topo_ref(
        &self,
    ) -> Option<&crate::exec::gpu::cuda::ExecViscousInteriorTopology> {
        self.cuda_viscous_topo.as_ref()
    }
}
