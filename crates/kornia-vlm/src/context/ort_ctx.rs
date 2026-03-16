use std::sync::Arc;
use kornia_image::{allocator::ImageAllocator, Image};
use tokenizers::Tokenizer;

// Assuming your backend path is correct, and we use the 'Device' enum you just wrote
use crate::backend::onnx::{OnnxEngine, Device}; 

pub enum ModelFamily {
    Paligemma,
    // QwenVL,
    // SmolVLM,
}

pub struct OrtContext {
    pub vision_engine: Arc<OnnxEngine>,
    pub language_engine: Arc<OnnxEngine>,
    pub tokenizer: Arc<Tokenizer>,
    pub family: ModelFamily,
}

impl OrtContext {
    pub fn new(vision_path: &str, lang_path: &str, tokenizer_path: &str, family: ModelFamily) -> Result<Self, String> {
        
        OnnxEngine::init_env().map_err(|e| e.to_string())?;
        
        let vision_engine = OnnxEngine::load(vision_path, Device::Cpu)
            .map_err(|e| e.to_string())?;
            
        let language_engine = OnnxEngine::load(lang_path, Device::Cuda { device_id: 0 })
            .map_err(|e| e.to_string())?;
            
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| e.to_string())?;

        Ok(Self { 
            vision_engine: Arc::new(vision_engine), 
            language_engine: Arc::new(language_engine), 
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
                
                let image_features = crate::paligemma::ort::vision::execute_vision(
                    &self.vision_engine, 
                    &f16_tensor
                ).map_err(|e| e.to_string())?;
                
                let response = crate::paligemma::ort::text::generate(
                    &self.language_engine,
                    &self.tokenizer,
                    image_features,
                    &formatted_prompt
                ).map_err(|e| e.to_string())?;
                
                Ok(response)
            }
        }
    }
}