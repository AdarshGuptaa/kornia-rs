use std::sync::Arc;
use std::sync::Mutex;
use std::path::PathBuf;
use kornia_image::{allocator::ImageAllocator, Image};
use tokenizers::Tokenizer;
use crate::backend::onnx::{OnnxEngine, Device};

// All the (implemented)models that use Onnx 
pub enum ModelFamily {
    Paligemma,
    // QwenVL,
    // SmolVLM,
}

pub struct OrtContext {
    pub vision_engine: Arc<Mutex<OnnxEngine>>,
    pub language_engine: Arc<Mutex<OnnxEngine>>,
    pub tokenizer: Arc<Tokenizer>,
    pub family: ModelFamily,
}

impl OrtContext {
    pub fn new(vision_path: PathBuf, lang_path: PathBuf, tokenizer_path: PathBuf, family: ModelFamily) -> Result<Self, String> {
        
        OnnxEngine::init_env().map_err(|e| e.to_string())?;
        
        let vision_engine = OnnxEngine::load(vision_path, Device::Cpu)
            .map_err(|e| e.to_string())?;
            
        let language_engine = OnnxEngine::load(lang_path, Device::Cpu)
            .map_err(|e| e.to_string())?;
            
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| e.to_string())?;

        Ok(Self { 
            vision_engine: Arc::new(Mutex::new(vision_engine)),
            language_engine: Arc::new(Mutex::new(language_engine)),
            tokenizer: Arc::new(tokenizer), 
            family 
        })
    }

    pub fn process_frame<A: ImageAllocator>(&self, image: &Image<u8, 3, A>, prompt: &str) -> Result<String, String> {
        match self.family {
            ModelFamily::Paligemma => {
                let f16_tensor = crate::paligemma::ort::preprocessor::process_image(image)
                    .map_err(|e| e.to_string())?;
                    
                let formatted_prompt = crate::paligemma::ort::preprocessor::process_text(prompt);
                
                let mut vision_guard = self.vision_engine.lock()
                    .map_err(|_| "Failed to lock vision engine mutex".to_string())?;

                let image_features = crate::paligemma::ort::vision::execute_vision(
                    &mut *vision_guard,
                    f16_tensor
                ).map_err(|e| e.to_string())?;
                
                let mut lang_guard = self.language_engine.lock()
                    .map_err(|_| "Failed to lock language engine mutex".to_string())?;

                let response = crate::paligemma::ort::text::generate(
                    &mut *lang_guard,
                    self.tokenizer.as_ref(),
                    image_features,
                    &formatted_prompt
                ).map_err(|e| e.to_string())?;
                
                Ok(response)    
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;
    use std::path::PathBuf;

    #[test]
    fn test_end_to_end_paligemma_pipeline() -> Result<(), String> {


        let mut tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
        tokenizer_path.pop(); 
        tokenizer_path.pop();
        tokenizer_path.push("tests");
        tokenizer_path.push("data");
        tokenizer_path.push("dummy_tokenizer.json");


        let mut vision_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
        vision_path.pop(); 
        vision_path.pop();
        vision_path.push("tests");
        vision_path.push("data");
        vision_path.push("vision_model.onnx");

        let mut lang_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
        lang_path.pop(); 
        lang_path.pop();
        lang_path.push("tests");
        lang_path.push("data");
        lang_path.push("language_model.onnx");

        println!("Initializing OrtContext...");
        let context = OrtContext::new(
            vision_path, 
            lang_path, 
            tokenizer_path, 
            ModelFamily::Paligemma
        )?;

        let dummy_size = ImageSize { width: 1280, height: 720 };
        let dummy_image = Image::<u8, 3, _>::from_size_val(dummy_size, 0, CpuAllocator)
            .map_err(|e| format!("Failed to create dummy image: {:?}", e))?;

        let prompt = "Describe this image in detail.";

        println!("Processing frame...");
        let result = context.process_frame(&dummy_image, prompt)?;

        println!("End-to-End Generation Success!\nOutput: {}", result);
        
        assert!(!result.is_empty(), "The generated response was empty!");

        Ok(())
    }
}