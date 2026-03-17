use ndarray::Array4;
use half::f16;
use crate::backend::onnx::OnnxEngine;

// Run the onnx engine on a image's pixel values tensor to give tensor result
pub fn execute_vision(
    engine: &mut OnnxEngine,
    pixel_values: Array4<f16>,
) -> Result<ndarray::ArrayD<f16>, String> {
    
    // convert input to ort value tensor for onnx engine to interpret
    let ort_input = ort::value::Tensor::from_array(pixel_values)
        .map_err(|e| e.to_string())?;

    // run onxx engine
    let outputs = engine.session.run(ort::inputs![
        "pixel_values" => ort_input
    ]).map_err(|e| e.to_string())?;
    
    // extract output to vec
    let extracted = outputs["image_features"] 
        .try_extract_tensor::<f16>()
        .map_err(|e| e.to_string())?;

    let shape_usize: Vec<usize> = extracted.0
        .iter()
        .map(|&dim| dim as usize)
        .collect();

    // Build result as a 4d Tensor in ndarray::Array4
    let image_features = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&shape_usize), 
        extracted.1.to_vec()
    ).map_err(|e| e.to_string())?;

    Ok(image_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    use half::f16;
    use crate::backend::onnx::{OnnxEngine, Device};
    use std::path::PathBuf;

    #[test]
    fn test_execute_vision_success() {
        let _ = OnnxEngine::init_env();

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop(); 
        path.pop();
        path.push("tests");
        path.push("data");
        path.push("vision_model.onnx");

        let mut engine = OnnxEngine::load(path, Device::Cpu)
            .expect("Failed to load dummy vision model");

        let pixel_values = Array4::<f16>::from_elem((1, 3, 224, 224), f16::from_f32(0.0));

        let result = execute_vision(&mut engine, pixel_values);

        assert!(
            result.is_ok(),
            "execute_vision failed with error: {:?}",
            result.err()
        );
        let image_features = result.unwrap();
        assert_eq!(
            image_features.shape(),
            &[1, 256, 2048],
            "Output tensor shape does not match expected PaliGemma dimensions"
        );
        assert_eq!(
            image_features[[0, 0, 0]],
            f16::from_f32(0.5f32),
            "Output tensor values do not match the expected dummy payload"
        );
    }
}