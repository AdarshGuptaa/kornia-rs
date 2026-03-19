use half::f16;

pub trait ModelFloat: Send + Sync + Copy + Default + PartialOrd + 'static  {
    fn from_f32_normalized(val: f32)-> Self;
    fn from_f16(val: f16) -> Self;
    fn neg_infinity() -> Self;
}

impl ModelFloat for f32{
    #[inline(always)]
    fn from_f32_normalized(val: f32)-> Self {
        val
    }
    #[inline(always)]
    fn from_f16(val: f16) -> Self { val.to_f32() } 
    #[inline(always)]
    fn neg_infinity() -> Self { f32::NEG_INFINITY }
}

impl ModelFloat for f16{    
    #[inline(always)]
    fn from_f32_normalized(val: f32)-> Self {
        f16::from_f32(val)
    }
    #[inline(always)]
    fn from_f16(val: f16) -> Self { val }
    #[inline(always)]
    fn neg_infinity() -> Self { f16::NEG_INFINITY }
}