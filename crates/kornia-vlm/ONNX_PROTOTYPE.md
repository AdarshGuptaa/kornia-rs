# Kornia-VLM: Thread-Safe ONNX Runtime Backend (Prototype)
This repository contains the Proof of Concept (PoC) for integrating an ONNX runtime Engine for VLMs such as Paligemma.
It provides a thread-safe and WIP ONNX Runtime backend designed to eventually ingest real-time video frames from the Bubbaloop application.

##  What Has Been Created
**The Pipeline:**
1. Initialize a 3-engine `OrtContext` (Vision, Embedding, Language)
2. Ingest raw images and text prompts via `process_frame()`
3. Preprocess images into natively aligned `ndarray` formats
4. Execute the Vision session to extract spatial `image_features`
5. Execute the Embedding session to project text into vectors
6. Stitch the tensors into a unified `inputs_embeds` sequence
7. Execute the autoregressive decoding loop to stream generated text

**Key Achievements:**
* `Candle` Implementation has been seperated using `[cfg]` feature flag: Safely preserving the candle implementations
* The Ort backend and Context have been built with reuse in mind: Future VLM implementations can always use the same backend and features.
* 3-Engine Architecture: Decoupled the embedding layer to natively support web-optimized `onnx-community` merged decoders.
* Dynamic Precision: Engineered `f32` and `f16` support via a custom `ModelFloat` trait. This slashes memory consumption in half, enabling inference on consumer GPUs.
* Thread-Safe Context Manager: Implemented `OrtContext`, a struct that safely manages ONNX sessions on concurrent threads.
* Autoregressive Loop: Built a greedy-search generation loop that feeds predicted token IDs back into the language decoder while 
  maintaining a static view of the image features. (Unoptimized)
* Integration Testing: An end-to-end `cfg(test)` pipeline that processes a dummy image through the entire vision and language architecture.

The backend is built on the `Arc<OnnxEngine>` pattern:
1. `Arc` (Atomic Reference Counted): Ensures that the massive model weights loaded into the `OnnxEngine` are shared safely across the application without copying data.
2. Thread-Safe Inference: The underlying `ort::Session` is inherently thread-safe (`Sync`). When a video frame arrives, the pipeline can run concurrent inferences using immutable references (`&self`), preventing data races and avoiding the severe bottlenecks that would be caused by `Mutex` locking during real-time execution.
3. Zero-Copy Views: During the 50-step token generation loop, the system passes borrowed views (`.view()`) of the image features rather than cloning the massive vision tensor on every iteration.

## Proof of Concept:
Image:
![umbrella](https://github.com/user-attachments/assets/7c292846-beb9-4b59-a061-dfcf6115060e)

Prompt: **caption en**

Code:
```
use std::path::PathBuf;
use kornia_vlm::backend::onnx::{Device, OnnxEngine};
use kornia_io::functional as F;
use kornia_vlm::context::ort_ctx::{ModelFamily, OrtContext};
use kornia_image::{Image, ImageSize, allocator::CpuAllocator};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = "/home/adarsh_gupta/Pictures/umbrella.jpeg";
    let model_base = PathBuf::from("/home/adarsh_gupta/Projects/paligemma_models/");
    let vision_path = model_base.join("vision_encoder_quantized.onnx");
    let lang_path = model_base.join("decoder_model_merged_quantized.onnx");
    let tokenizer_path = model_base.join("/home/adarsh_gupta/Projects/paligemma_models/tokenizer.json");
    let embedds_path = model_base.join("/home/adarsh_gupta/Projects/paligemma_models/embed_tokens_quantized.onnx");
    OnnxEngine::<()>::init_env()
    .map_err(|e| format!("Environment initialization failed: {}", e))?;
    let image = F::read_image_any_rgb8(image_path)
    .map_err(|e| format!("Failed to read image: {:?}", e))?;
    let context = OrtContext::<f32>::new(
        vision_path,
        embedds_path,
        lang_path,
        tokenizer_path,
        ModelFamily::Paligemma,
    )?;
    let prompt = "caption en";
    println!("Generating caption...");
    match context.process_frame(&image, prompt) {
        Ok(caption) => {
            println!("------------------------------------");
            println!("Model Output: {}", caption);
            println!("------------------------------------");
        }
        Err(e) => eprintln!("Inference Error: {}", e),
    }
    Ok(())
}
```
Output:

<img width="606" height="347" alt="Screenshot_20260319_150313" src="https://github.com/user-attachments/assets/a3adb9f6-45b6-41a8-8303-5c3e0fc3768e" />

This prototype relies on f32 i64 quantized encoder, decoder, tokenizer and token embedder from https://huggingface.co/onnx-community/paligemma2-3b-pt-224/tree/main

**Decoder**: decoder_model_merged_quantized.onnx + decoder_model_merged_quantized.onnx_data
**Encoder**: encoder_model_merged_quantized.onnx
**Tokenizer**: tokenizer.json + tokenizer_config.json
**Token Embeds**: embed_tokens_quantized.onnx

## TODOs & Next Steps
The following steps are required to transition this prototype into a production-ready feature for Bubbaloop:
- [ ] Generalize the implementation for all paligemma models by handling their specific `config.json` files for multimodel robustness.
- [ ] Bubbaloop Integration: Connect the `OrtContext::process_frame` function into the `main.rs` Bubbaloop camera ingestion loop.
- [x] Real Model Weights: Swap the randomly initialized dummy ONNX models for actual, quantized exported weights (e.g., Qwen2.5-VL or PaliGemma).
- [ ] Execution Providers: Explicitly configure `ort` to utilize hardware acceleration (TensorRT / CUDA) during session initialization for real-time FPS.
- [ ] KV Caching: Implement Key-Value caching in the autoregressive loop to prevent recalculating past tokens, drastically speeding up text generation.
- [ ] Dynamic Resizing: Ensure the preprocessor can handle arbitrary, non-square webcam resolutions and convert to model multimodel specific formats.
- [ ] Error Handling: Create appropriate errors using ort::Error and thisError crate
- [ ] Handle preprocessing, input_ids production and the autoregressive loop using SIMD/Candle_core/Kornia-Tensor(Working with project-1 contributor for GSOC)
