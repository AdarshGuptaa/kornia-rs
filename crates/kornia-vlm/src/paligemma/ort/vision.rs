use ndarray::Array4;
use half::f16;
use crate::backend::onnx::OnnxEngine;

pub fn execute_vision(
    engine: &OnnxEngine,
    pixel_values: &Array4<f16>,
) -> Result<ndarray::ArrayD<f32>, String> {
    
    let outputs = engine.session.run(ort::inputs![
        "pixel_values" => pixel_values.view()
    ].map_err(|e: ort::Error| e.to_string())?)
    .map_err(|e: ort::Error| e.to_string())?;

    let extracted = outputs["image_features"] 
        .try_extract_tensor::<f32>()
        .map_err(|e: ort::Error| e.to_string())?;

    let shape_usize: Vec<usize> = extracted.0
        .iter()
        .map(|&dim| dim as usize)
        .collect();
    
    let image_features = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&shape_usize), 
        extracted.1.to_vec()
    ).map_err(|e| e.to_string())?;

    Ok(image_features)
}