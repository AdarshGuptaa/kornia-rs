# Kornia-VLM: Thread-Safe ONNX Runtime Backend (Prototype)
This repository contains the Proof of Concept (PoC) for integrating an ONNX runtime Engine for VLMs such as Paligemma.
It provides a thread-safe and WIP ONNX Runtime backend designed to eventually ingest real-time video frames from the Bubbaloop application.

##  What Has Been Created
A complete, end-to-end inference pipeline for VLMs (like PaliGemma and Qwen) using `ort v2.0.0-rc11`.
Pipeline works as follows:
1. Onnx Context (High Level API) new instance created
2. OnnxEngine is initiated
3. Vision and Text Model are loaded in the form of sessions in the engine
4. Tokenizer is loaded into engine environment
5. Image and Prompt are loaded through context::process_frame()
6. Image is processed to the format acceptable to Paligemma Model
7. Processed image is run on vision.rs to produce image_features
8. text.rs takes image_features and text prompt to produce text tokens and run the autoregressive loop to produce final string output

**Key Achievements:**
* Thread-Safe Context Manager: Implemented `OrtContext`, a struct that safely manages ONNX sessions on concurrent threads.
* Autoregressive Loop: Built a greedy-search generation loop that feeds predicted token IDs back into the language decoder while 
  maintaining a static view of the image features. (Unoptimized)
* Integration Testing: An end-to-end `cfg(test)` pipeline that processes a dummy image through the entire vision and language architecture.

The backend is built on the `Arc<OnnxEngine>` pattern:
1. `Arc` (Atomic Reference Counted): Ensures that the massive model weights loaded into the `OnnxEngine` are shared safely across the application without copying data.
2. Thread-Safe Inference: The underlying `ort::Session` is inherently thread-safe (`Sync`). When a video frame arrives, the pipeline can run concurrent inferences using immutable references (`&self`), preventing data races and avoiding the severe bottlenecks that would be caused by `Mutex` locking during real-time execution.
3. Zero-Copy Views: During the 50-step token generation loop, the system passes borrowed views (`.view()`) of the image features rather than cloning the massive vision tensor on every iteration.

## Proof of Concept:
Run the following:
- `src/backend/onnx.rs` tests
- `src/context/onnx_ctx.rs` tests (Complete high Level Test)
- `src/paligemma/text.rs|vision.rs|preprocessor.rs` tests

This prototype relies on decoupled, randomly initialized dummy ONNX models.
Both models use **FP16 (Float16)** precision to be in check with the edge inference standards (For Jetson)

**Vision Model**: SigLIP mock
* Input (`pixel_values`):** `[batch_size, 3, 224, 224]` (FP16).
* Output (`image_features`):** `[batch_size, 256, 2048]` (FP16).

**Language Model**: Gemma mock
* Inputs:  `inputs_embeds`: `[batch_size, seq_len, 2048]` (FP16).
          `attention_mask`: `[batch_size, seq_len]` (INT64).
* Output (`logits`): `[batch_size, seq_len, 257152]` (FP16).


## TODOs & Next Steps
The following steps are required to transition this prototype into a production-ready feature for Bubbaloop:
- [ ] Bubbaloop Integration: Connect the `OrtContext::process_frame` function into the `main.rs` Bubbaloop camera ingestion loop.
- [ ] Real Model Weights: Swap the randomly initialized dummy ONNX models for actual, quantized exported weights (e.g., Qwen2.5-VL or PaliGemma).
- [ ] Execution Providers: Explicitly configure `ort` to utilize hardware acceleration (TensorRT / CUDA) during session initialization for real-time FPS.
- [ ] KV Caching: Implement Key-Value caching in the autoregressive loop to prevent recalculating past tokens, drastically speeding up text generation.
- [ ] Dynamic Resizing: Ensure the preprocessor can handle arbitrary, non-square webcam resolutions and convert to model multimodel specific formats.
- [ ] Error Handling: Create appropriate errors using ort::Error and thisError crate
- [ ] Handle preprocessing, input_ids production and the autoregressive loop using SIMD/Candle_core/Kornia-Tensor(Working with project-1 contributor for GSOC)
