use kornia_image::{allocator::ImageAllocator, Image, ImageSize, allocator::CpuAllocator};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};
use ndarray::Array4;
use half::f16;

// fit an image to the paligemma requirements
// TODO: Optimise for SIMD or GPU 
pub fn process_image<A: ImageAllocator>(image: &Image<u8, 3, A>) -> Result<Array4<f16>, String> {

    // TODO: Refactor to handle 224, 448 and 896 resolutions
    let new_size = ImageSize {
        width: 224,
        height: 224,
    };

    let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0u8, CpuAllocator).unwrap();

    // TODO: Remove unwrap
    resize_fast_rgb(
        &image, 
        &mut image_resized,
        InterpolationMode::Bilinear
    ).unwrap();

    let height = 224;
    let width = 224;
    let channels = 3;
    let mut chw_buffer = vec![f16::ZERO; channels * height * width];
    let plane_size = height * width;
    let hwc_data = image_resized.as_slice();

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

#[cfg(test)]
mod tests {
    use kornia_image::{Image, allocator::CpuAllocator, ImageSize};
    use half::f16;
    use crate::paligemma::ort::preprocessor::process_image;

    #[test]
    fn test_process_image_shape_and_normalization() {
        let original_width = 300;
        let original_height = 300;
        let white_pixels = 255u8;

        let dummy_image = Image::<u8, 3, _>::from_size_val(
            ImageSize { width: original_width, height: original_height }, 
            white_pixels,
            CpuAllocator
        ).expect("Failed to create dummy image for testing");

        let result = process_image(&dummy_image);

        assert!(result.is_ok(), "process_image failed: {:?}", result.err());
        let tensor = result.unwrap();

        assert_eq!(tensor.shape(), &[1, 3, 224, 224], "Tensor shape is incorrect!");

        let expected_f16 = f16::from_f32(1.0);

        assert_eq!(tensor[[0, 0, 0, 0]], expected_f16, "Red channel top-left pixel failed");
        assert_eq!(tensor[[0, 1, 112, 112]], expected_f16, "Green channel center pixel failed");
        assert_eq!(tensor[[0, 2, 223, 223]], expected_f16, "Blue channel bottom-right pixel failed");
    }

    #[test]
    fn test_process_image_black_normalization() {

        let black_pixels = 0u8;
        
        let dummy_image = Image::<u8, 3, _>::from_size_val(
            ImageSize { width: 300, height: 300 }, 
            black_pixels,
            CpuAllocator
        ).unwrap();

        let tensor = process_image(&dummy_image).unwrap();
        
        let expected_f16 = f16::from_f32(-1.0);
        assert_eq!(tensor[[0, 0, 10, 10]], expected_f16, "Black pixel normalization failed");
    }
}