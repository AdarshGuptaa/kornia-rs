use ort::{
    session::builder::GraphOptimizationLevel,
    session::{Session},
    execution_providers::CUDAExecutionProvider,
    memory::Allocator, 
};
use std::path::Path;

pub enum Device {
    Cpu,
    Cuda { device_id: i32 },
}

pub struct OnnxEngine {
    pub session: Session,
    pub input_names: Vec<String>,
}

// TODO: Handle multiple OnnxEngine sessions concurrently
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
            .with_intra_threads(4)?; // TODO: optimize thread usage based on the system

        // Cuda and Cpu(Default)
        // TODO: TensorRT impl
        // TODO: Cuda fallback cascade
        builder = match device { 
            Device::Cuda { device_id } => {
                builder.with_execution_providers([
                    CUDAExecutionProvider::default().with_device_id(device_id).build()
                ])?
            },
            Device::Cpu => builder,
        };

        let session = builder.commit_from_file(model_path)?;

        // fetches the name of the inputs in the loaded model
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();

        Ok(Self { session, input_names})
    }

    // exposes the allocator for rust references -> c++ values
    pub fn allocator(&self) -> &Allocator {
        self.session.allocator()
    }

}

#[cfg(test)]
mod tests {
    use crate::backend::onnx::{OnnxEngine, Device};
    use half::f16;
    use std::path::PathBuf;


    #[test]
fn test_engine_lifecycle_and_inference() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the global environment
    OnnxEngine::init_env().map_err(|e| format!("Env Init Failed: {}", e))?;

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("tests");
    path.push("data");
    path.push("vision_model.onnx");

    assert!(path.exists(), "Still not found at {:?}", path);

    // load the onnx engine
    let mut engine = OnnxEngine::load(path, Device::Cpu)?;
    
    let dummy_input = ndarray::Array4::<f16>::zeros((1, 3, 224, 224));
    
    // convert input to an ort value for the engine to understand
    let ort_input = ort::value::Tensor::from_array(dummy_input)
    .map_err(|e| e.to_string())?;

    let inputs = ort::inputs![
        "pixel_values" => &ort_input
    ];

    // Run on the engine
    let outputs = engine.session.run(inputs)?;

    // extract the output as vec
    let image_features = outputs["image_features"]
        .try_extract_tensor::<f16>()?;
    
    assert_eq!(image_features.0.len(), 3); // check rank
    assert_eq!(image_features.0[0], 1);    // check batch size

    Ok(())
}
}