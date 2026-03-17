use ndarray::{Array2, ArrayD};
use tokenizers::Tokenizer;
use crate::backend::onnx::OnnxEngine;
use half::f16;

pub fn generate(
    engine: &mut OnnxEngine,
    tokenizer: &Tokenizer,
    image_features: ArrayD<f16>,
    prompt: &str,
) -> Result<String, String> {
    
    // divided into tokens
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| e.to_string())?;
    
    // token ids in vector (Onnx only takes i64 token ids)
    let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let eos_token_id = tokenizer.token_to_id("<eos>").unwrap_or(1) as i64;
    
    let mut generated_text = String::new();

    // process 50 tokens at a time
    let max_tokens = 50;

    let image_features_val = ort::value::Tensor::from_array(image_features)
            .map_err(|e| e.to_string())?;


    let input_names = &engine.input_names;

    let text_input_name = &input_names[1];
    let image_input_name = &input_names[0];

    // auto regressive loop
    for _ in 0..max_tokens {

        let seq_len = input_ids.len();
        
        // reshape the token ids vector into a 2d ndarray vector
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids.clone())
            .map_err(|e| e.to_string())?;

        let input_ids_val = ort::value::Tensor::from_array(input_ids_array)
            .map_err(|e| e.to_string())?;
        
        // run the tokens through the onnx backend
        let outputs = engine.session.run(ort::inputs![
            text_input_name.as_str() => input_ids_val,
            image_input_name.as_str() => &image_features_val,
        ]).map_err(|e: ort::Error| e.to_string())?;
        
        // extract output
        let extracted = outputs["logits"]
            .try_extract_tensor::<f16>()
            .map_err(|e: ort::Error| e.to_string())?;

        let logits_shape = extracted.0;
        let logits_slice = extracted.1;
        
        let vocab_size = *logits_shape.last().unwrap_or(&1) as usize;

        // get start score of last token
        let last_token_start = logits_slice.len() - vocab_size;

        // get the lask token as vec of logits
        let last_token_logits = &logits_slice[last_token_start..];
        
        let mut best_token_id = 0;
        let mut highest_prob = f16::NEG_INFINITY;
        
        // greedy loop
        for (id, &prob) in last_token_logits.iter().enumerate() {
            if prob > highest_prob {
                highest_prob = prob;
                best_token_id = id as i64;
            }
        }

        // Model is done with its text
        if best_token_id == eos_token_id { break; }

        input_ids.push(best_token_id);

        if let Some(word) = tokenizer.decode(&[best_token_id as u32], true).ok() {
            generated_text.push_str(&word);
        }
    }

    Ok(generated_text.trim().to_string())
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::backend::onnx::{Device, OnnxEngine};
    use half::f16;
    use std::path::PathBuf;

    #[test]
    fn test_autoregressive_gen(){
        let _ = OnnxEngine::init_env();

        let mut tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
        // Navigate from crates/kornia-vlm -> crates/ -> workspace_root
        tokenizer_path.pop(); 
        tokenizer_path.pop();
        tokenizer_path.push("tests");
        tokenizer_path.push("data");
        tokenizer_path.push("dummy_tokenizer.json");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .expect("Failed to load tokenizer");

        let mut model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
        // Navigate from crates/kornia-vlm -> crates/ -> workspace_root
        model_path.pop(); 
        model_path.pop();
        model_path.push("tests");
        model_path.push("data");
        model_path.push("language_model.onnx");

        let mut engine = OnnxEngine::load(model_path, Device::Cpu).unwrap();

        let dummy_img_shape = vec![1, 64, 2048];
        let total_elements = dummy_img_shape.iter().product();

        let dummy_img: Vec<f16> = vec![f16::from_f32(0.0); total_elements];
        let image_features = ArrayD::from_shape_vec(dummy_img_shape, dummy_img)
            .expect("Failed to create a dummy image feature");

        let prompt = "Describe the Image";

        let result = generate(&mut engine, &tokenizer, image_features, prompt);

        assert!(result.is_ok(), "Generation failed with error: {:?}", result.err());

        let generated_text = result.unwrap();
        println!("Generated text: {}", generated_text);

        assert!(!generated_text.is_empty(), "Generated text should not be empty");

    }
}