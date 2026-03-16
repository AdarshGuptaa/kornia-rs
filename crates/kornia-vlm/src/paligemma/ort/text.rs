use ndarray::{Array2, ArrayD};
use tokenizers::Tokenizer;
use crate::backend::onnx::OnnxEngine;

pub fn generate(
    engine: &OnnxEngine,
    tokenizer: &Tokenizer,
    image_features: ArrayD<f32>,
    prompt: &str,
) -> Result<String, String> {
    
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| e.to_string())?;
    
    let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let eos_token_id = tokenizer.token_to_id("<eos>").unwrap_or(1) as i64;
    
    let mut generated_text = String::new();
    let max_tokens = 50; 

    for _ in 0..max_tokens {
        let seq_len = input_ids.len();
        
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids.clone())
            .map_err(|e| e.to_string())?;

        let outputs = engine.session.run(ort::inputs![
            "input_ids" => input_ids_array.view(),
            "image_features" => image_features.view()
        ].map_err(|e: ort::Error| e.to_string())?)
        .map_err(|e: ort::Error| e.to_string())?; 

        let extracted = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| e.to_string())?;

        let logits_shape = extracted.0;
        let logits_slice = extracted.1;

        let vocab_size = *logits_shape.last().unwrap_or(&1) as usize;
        let last_token_start = logits_slice.len() - vocab_size;
        let last_token_logits = &logits_slice[last_token_start..];
        
        let mut best_token_id = 0;
        let mut highest_prob = f32::NEG_INFINITY;
        
        for (id, &prob) in last_token_logits.iter().enumerate() {
            if prob > highest_prob {
                highest_prob = prob;
                best_token_id = id as i64;
            }
        }

        if best_token_id == eos_token_id { break; }

        input_ids.push(best_token_id);

        if let Some(word) = tokenizer.decode(&[best_token_id as u32], true).ok() {
            generated_text.push_str(&word);
        }
    }

    Ok(generated_text.trim().to_string())
}