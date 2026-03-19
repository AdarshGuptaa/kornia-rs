use crate::backend::onnx::OnnxEngine;
use ndarray::{Array2, Array3, ArrayD, Axis};
use ort::tensor::PrimitiveTensorElementType;
use std::fmt::Debug;
use tokenizers::Tokenizer;

pub fn generate<T>(
    decoder_engine: &mut OnnxEngine<T>,
    embed_engine: &mut OnnxEngine<T>, 
    tokenizer: &Tokenizer,
    image_features: ArrayD<T>,        
    input_ids_array: Array2<i64>,     
) -> Result<String, String>
where
    T: crate::types::ModelFloat + PrimitiveTensorElementType + Debug + PartialOrd,
{
    let mut input_ids = input_ids_array.into_raw_vec();
    let eos_token_id = tokenizer.token_to_id("<eos>").unwrap_or(1) as i64;
    let mut generated_text = String::new();
    let max_tokens = 50;
    let num_layers = 26;

    // no kv cache loop
    for _ in 0..max_tokens {
        // Rebuild the full Image + Text sandwich every step :(
        let current_embeds_array = build_prefill_embeddings(
            embed_engine, 
            image_features.clone(), // Clone so we don't consume the image
            &input_ids
        )?;

        let seq_len = current_embeds_array.shape()[1]; 
        
        let position_ids: Vec<i64> = (0..seq_len as i64).collect();
        let position_tensor = ort::value::Tensor::from_array(
            ndarray::Array2::from_shape_vec((1, seq_len), position_ids).unwrap()
        ).unwrap();

        let full_embeds_tensor = ort::value::Tensor::from_array(current_embeds_array).unwrap();

        let mut session_inputs: Vec<(&str, ort::value::DynValue)> = vec![
            ("position_ids", position_tensor.into()),
            ("inputs_embeds", full_embeds_tensor.into()),
        ];

        // Feed 52 Empty Tensors every single time 
        for i in 0..num_layers {
            let empty_kv = ndarray::Array4::<f32>::zeros((1, 4, 0, 256));
            session_inputs.push((Box::leak(format!("past_key_values.{}.key", i).into_boxed_str()), ort::value::Tensor::from_array(empty_kv.clone()).unwrap().into()));
            session_inputs.push((Box::leak(format!("past_key_values.{}.value", i).into_boxed_str()), ort::value::Tensor::from_array(empty_kv).unwrap().into()));
        }

        // Run decoder egine
        let outputs = decoder_engine.session.run(session_inputs).map_err(|e: ort::Error| e.to_string())?;
        
        // Extract Logits
        let extracted = outputs["logits"].try_extract_tensor::<T>().map_err(|e: ort::Error| e.to_string())?;
        let (logits_shape, logits_slice) = (extracted.0, extracted.1);
        let vocab_size = *logits_shape.last().unwrap_or(&1) as usize;

        let last_token_logits = &logits_slice[logits_slice.len() - vocab_size..];
        
        let mut best_token_id = 0;
        let mut highest_prob = T::neg_infinity();
        for (id, &prob) in last_token_logits.iter().enumerate() {
            if prob > highest_prob {
                highest_prob = prob;
                best_token_id = id as i64;
            }
        }

        if best_token_id == eos_token_id { break; }

        input_ids.push(best_token_id);
        
        if let Some(word) = tokenizer.decode(&[best_token_id as u32], false).ok() {
            generated_text.push_str(&word);
            println!("Generated: {}", word); // Print as it thinks!
        }
    }

    Ok(generated_text.trim().to_string())
}

fn build_prefill_embeddings<T>(
    embed_engine: &mut OnnxEngine<T>,
    image_features: ArrayD<T>,
    prompt_ids: &[i64],
) -> Result<Array3<T>, String>
where
    T: crate::types::ModelFloat + PrimitiveTensorElementType + Debug,
{
    // Process text through the embedding model
    let prompt_tensor = ort::value::Tensor::from_array(
        Array2::from_shape_vec((1, prompt_ids.len()), prompt_ids.to_vec()).unwrap(),
    )
    .map_err(|e| e.to_string())?;

    let embed_outputs = embed_engine
        .session
        .run(ort::inputs!["input_ids" => prompt_tensor])
        .map_err(|e: ort::Error| e.to_string())?;

    // Extract and clone to break the borrow
    let (text_shape, text_data) = embed_outputs["inputs_embeds"]
        .try_extract_tensor::<T>()
        .map_err(|e: ort::Error| e.to_string())?;

    let text_shape_usize: Vec<usize> = text_shape.iter().map(|&x| x as usize).collect();
    let text_embeds_3d = ndarray::Array::from_shape_vec(text_shape_usize, text_data.to_vec())
        .map_err(|e| format!("Failed to create Array: {}", e))?
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| "Failed to cast text_embeds to 3D array.")?;

    let image_features_3d = image_features
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| "Failed to cast image_features to 3D array.")?;

    // combine and return the array
    ndarray::concatenate(Axis(1), &[image_features_3d.view(), text_embeds_3d.view()])
        .map_err(|e| format!("Concatenation failed: {}", e))
}

/// Fetches the vector embedding for a single token ID.
fn embed_single_token<T>(
    embed_engine: &mut OnnxEngine<T>,
    token_id: i64,
) -> Result<Array3<T>, String>
where
    T: crate::types::ModelFloat + PrimitiveTensorElementType + Debug,
{
    let token_tensor = ort::value::Tensor::from_array(Array2::from_elem((1, 1), token_id))
        .map_err(|e| e.to_string())?;

    let embed_outputs = embed_engine
        .session
        .run(ort::inputs!["input_ids" => token_tensor])
        .map_err(|e: ort::Error| e.to_string())?;

    let (shape, data) = embed_outputs["inputs_embeds"]
        .try_extract_tensor::<T>()
        .map_err(|e: ort::Error| e.to_string())?;

    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

    ndarray::Array::from_shape_vec(shape_usize, data.to_vec())
        .map_err(|e| format!("Failed to rebuild embed array: {}", e))?
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| String::from("Failed to cast to 3D array."))
}
