use kornia_image::{allocator::ImageAllocator, Image, Size, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize};
use ndarray::Array4;
use half::f16;

/// Resizes and normalizes the image specifically for PaliGemma's SigLIP encoder.
pub fn process_image<A: ImageAllocator>(image: &Image<u8, 3, A>) -> Result<Array4<f16>, String> {
    
    let resized = resize(
        image, 
        Size { width: 224, height: 224 }, 
        InterpolationMode::Bilinear
    ).map_err(|e| e.to_string())?;

    let height = 224;
    let width = 224;
    let channels = 3;
    let mut chw_buffer = vec![f16::ZERO; channels * height * width];
    let plane_size = height * width;
    let hwc_data = resized.as_slice();

    for y in 0..height {
        for x in 0..width {
            let src_idx = (y * width + x) * channels;
            
            let r = (hwc_data[src_idx] as f32 / 255.0 - 0.5) / 0.5;
            let g = (hwc_data[src_idx + 1] as f32 / 255.0 - 0.5) / 0.5;
            let b = (hwc_data[src_idx + 2] as f32 / 255.0 - 0.5) / 0.5;

            let spatial_idx = y * width + x;
            chw_buffer[spatial_idx]                  = f16::from_f32(r);
            chw_buffer[plane_size + spatial_idx]     = f16::from_f32(g);
            chw_buffer[2 * plane_size + spatial_idx] = f16::from_f32(b);
        }
    }

    Array4::from_shape_vec((1, channels, height, width), chw_buffer)
        .map_err(|e| e.to_string())
}

/// PaliGemma requires the text prompt to be prefixed with a specific image token.
pub fn process_text(prompt: &str) -> String {
    format!("<image>\n{}", prompt)
}