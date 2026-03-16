use ort::{
    session::builder::GraphOptimizationLevel,
    session::{Session, SessionInputs, SessionOutputs},
    execution_providers::CUDAExecutionProvider,
    Allocator, 
};
use std::path::Path;

pub enum Device {
    Cpu,
    Cuda { device_id: i32 },
}

pub struct OnnxEngine {
    pub session: Session,
}

impl OnnxEngine {
    // initialises the Engine and its underlying C++ implementation
    pub fn init_env() -> ort::Result<()> {
        let _ = ort::init()
            .with_name("kornia-vlm-engine")
            .commit(); 
        Ok(())
    }

    // Loads a model as a session into the memory of the selected device
    pub fn load<P: AsRef<Path>>(model_path: P, device: Device) -> ort::Result<Self> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        builder = match device { 
            Device::Cuda { device_id } => {
                builder.with_execution_providers([
                    CUDAExecutionProvider::default().with_device_id(device_id).build()
                ])?
            },
            Device::Cpu => builder,
        };

        let session = builder.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    // exposes the allocator for rust references -> c++ values
    pub fn allocator(&self) -> &Allocator {
        self.session.allocator()
    }
}