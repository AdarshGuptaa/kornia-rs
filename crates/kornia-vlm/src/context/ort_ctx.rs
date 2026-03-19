use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;
use std::path::PathBuf;
use kornia_image::{allocator::ImageAllocator, Image};
use ort::tensor::PrimitiveTensorElementType;
use tokenizers::Tokenizer;
use crate::backend::onnx::{OnnxEngine, Device};
use crate::types::ModelFloat;

// All the (implemented) models that use Onnx 
pub enum ModelFamily {
    Paligemma,
    // QwenVL,
    // SmolVLM,
}

pub struct OrtContext<T> {
    pub vision_engine: Arc<Mutex<OnnxEngine<T>>>,
    pub embed_engine: Arc<Mutex<OnnxEngine<T>>>,
    pub language_engine: Arc<Mutex<OnnxEngine<T>>>,
    pub tokenizer: Arc<Tokenizer>,
    pub family: ModelFamily,
}

impl<T> OrtContext<T>
where
    T: ModelFloat + PrimitiveTensorElementType + Debug + PartialOrd
{
    pub fn new(
        vision_path: PathBuf, 
        embed_path: PathBuf,
        lang_path: PathBuf, 
        tokenizer_path: PathBuf, 
        family: ModelFamily
    ) -> Result<Self, String> {
        
        OnnxEngine::<()>::init_env().map_err(|e| e.to_string())?;
        
        let vision_engine = OnnxEngine::<T>::load(vision_path, Device::Cpu)
            .map_err(|e| e.to_string())?;
            
        // Load the new embedding engine
        let embed_engine = OnnxEngine::<T>::load(embed_path, Device::Cpu)
            .map_err(|e| e.to_string())?;

        let language_engine = OnnxEngine::<T>::load(lang_path, Device::Cpu)
            .map_err(|e| e.to_string())?;
            
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| e.to_string())?;

        Ok(Self { 
            vision_engine: Arc::new(Mutex::new(vision_engine)),
            embed_engine: Arc::new(Mutex::new(embed_engine)),
            language_engine: Arc::new(Mutex::new(language_engine)),
            tokenizer: Arc::new(tokenizer), 
            family 
        })
    }

    pub fn process_frame<A>(&self, image: &Image<u8, 3, A>, prompt: &str) -> Result<String, String>
    where
        A: ImageAllocator,
    {
        match self.family {
            ModelFamily::Paligemma => {
                let tensor = crate::paligemma::ort::preprocessor::process_image::<T, A>(image)
                    .map_err(|e| e.to_string())?;
                    
                let input_ids = crate::paligemma::ort::preprocessor::encode_prompt(
                    self.tokenizer.as_ref(), 
                    prompt
                )?;
                
                let mut vision_guard = self.vision_engine.lock()
                    .map_err(|_| "Failed to lock vision engine mutex".to_string())?;

                let image_features = crate::paligemma::ort::vision::execute_vision::<T>(
                    &mut *vision_guard,
                    tensor
                ).map_err(|e| e.to_string())?;
                
                // Lock the embedding engine
                let mut embed_guard = self.embed_engine.lock()
                    .map_err(|_| "Failed to lock embed engine mutex".to_string())?;

                let mut lang_guard = self.language_engine.lock()
                    .map_err(|_| "Failed to lock language engine mutex".to_string())?;

                // Pass the embed_guard into the generate function
                let response = crate::paligemma::ort::text::generate(
                    &mut *lang_guard,
                    &mut *embed_guard, 
                    self.tokenizer.as_ref(),
                    image_features,
                    input_ids
                ).map_err(|e| e.to_string())?;
                
                Ok(response)    
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::context::ort_ctx::{OrtContext, ModelFamily};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;
    use std::path::PathBuf;
    use half::f16;

    #[test]
    fn test_end_to_end_paligemma_pipeline() -> Result<(), String> {

        let mut base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        base_path.pop(); 
        base_path.pop();
        base_path.push("tests");
        base_path.push("data");

        let tokenizer_path = base_path.join("dummy_tokenizer.json");
        let vision_path = base_path.join("vision_model.onnx");
        let embed_path = base_path.join("embed_model.onnx");
        let lang_path = base_path.join("language_model.onnx");

        println!("Initializing OrtContext...");
        let context: OrtContext<f16> = OrtContext::<f16>::new(
            vision_path, 
            embed_path,
            lang_path, 
            tokenizer_path, 
            ModelFamily::Paligemma
        )?;

        let dummy_size = ImageSize { width: 1280, height: 720 };
        let dummy_image = Image::<u8, 3, _>::from_size_val(dummy_size, 0, CpuAllocator)
            .map_err(|e| format!("Failed to create dummy image: {:?}", e))?;

        let prompt = "Describe this image in detail.";

        println!("Processing frame...");
        let result = OrtContext::<f16>::process_frame(&context, &dummy_image, prompt)?;

        println!("End-to-End Generation Success!\nOutput: {}", result);
        
        assert!(!result.is_empty(), "The generated response was empty!");

        Ok(())
    }
}