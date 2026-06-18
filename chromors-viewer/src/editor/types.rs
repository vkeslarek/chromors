use chromors::backend::gpu::GpuBackend;
use chromors::data::image::Image2D as GpuImage;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DataType {
    Image,
    Mask,
    Scalar,
    Color,
}

impl DataType {
    pub fn accepts(self, producer: DataType) -> bool {
        self == producer || (self == DataType::Image && producer == DataType::Mask)
    }
}

#[derive(Clone)]
pub enum PortValue {
    Image(GpuImage<GpuBackend>),
    Mask(GpuImage<GpuBackend>),
    Scalar(f64),
    Color([f32; 4]),
}

impl PortValue {
    pub fn data_type(&self) -> DataType {
        match self {
            PortValue::Image(_) => DataType::Image,
            PortValue::Mask(_) => DataType::Mask,
            PortValue::Scalar(_) => DataType::Scalar,
            PortValue::Color(_) => DataType::Color,
        }
    }

    pub fn image(&self) -> &GpuImage<GpuBackend> {
        match self {
            PortValue::Image(i) => i,
            PortValue::Mask(i) => i,
            _ => panic!("Expected Image PortValue"),
        }
    }

    pub fn scalar(&self) -> f64 {
        match self {
            PortValue::Scalar(s) => *s,
            _ => panic!("Expected Scalar PortValue"),
        }
    }

    pub fn color(&self) -> [f32; 4] {
        match self {
            PortValue::Color(c) => *c,
            _ => panic!("Expected Color PortValue"),
        }
    }
}
