#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Choice(usize),
    Color([f32; 4]),
    Path(Option<String>),
}

impl ParamValue {
    pub fn float(&self) -> f64 {
        match self {
            ParamValue::Float(f) => *f,
            _ => panic!("Expected Float ParamValue"),
        }
    }
    pub fn int(&self) -> i64 {
        match self {
            ParamValue::Int(i) => *i,
            _ => panic!("Expected Int ParamValue"),
        }
    }
    pub fn bool(&self) -> bool {
        match self {
            ParamValue::Bool(b) => *b,
            _ => panic!("Expected Bool ParamValue"),
        }
    }
    pub fn choice(&self) -> usize {
        match self {
            ParamValue::Choice(c) => *c,
            _ => panic!("Expected Choice ParamValue"),
        }
    }
    pub fn color(&self) -> [f32; 4] {
        match self {
            ParamValue::Color(c) => *c,
            _ => panic!("Expected Color ParamValue"),
        }
    }
    pub fn path(&self) -> Option<&str> {
        match self {
            ParamValue::Path(p) => p.as_deref(),
            _ => panic!("Expected Path ParamValue"),
        }
    }
}

pub enum ParamSpec {
    Float { name: &'static str, min: f64, max: f64, default: f64 },
    Int { name: &'static str, min: i64, max: i64, default: i64 },
    Bool { name: &'static str, default: bool },
    Choice { name: &'static str, options: &'static [&'static str], default: usize },
    Color { name: &'static str, default: [f32; 4] },
    Path { name: &'static str },
}

impl ParamSpec {
    pub fn float(name: &'static str, min: f64, max: f64, default: f64) -> Self {
        Self::Float { name, min, max, default }
    }
    pub fn int(name: &'static str, min: i64, max: i64, default: i64) -> Self {
        Self::Int { name, min, max, default }
    }
    pub fn bool(name: &'static str, default: bool) -> Self {
        Self::Bool { name, default }
    }
    pub fn choice(name: &'static str, options: &'static [&'static str], default: usize) -> Self {
        Self::Choice { name, options, default }
    }
    pub fn color(name: &'static str, default: [f32; 4]) -> Self {
        Self::Color { name, default }
    }
    pub fn path(name: &'static str) -> Self {
        Self::Path { name }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Float { name, .. } => name,
            Self::Int { name, .. } => name,
            Self::Bool { name, .. } => name,
            Self::Choice { name, .. } => name,
            Self::Color { name, .. } => name,
            Self::Path { name, .. } => name,
        }
    }
    pub fn default_value(&self) -> ParamValue {
        match self {
            Self::Float { default, .. } => ParamValue::Float(*default),
            Self::Int { default, .. } => ParamValue::Int(*default),
            Self::Bool { default, .. } => ParamValue::Bool(*default),
            Self::Choice { default, .. } => ParamValue::Choice(*default),
            Self::Color { default, .. } => ParamValue::Color(*default),
            Self::Path { .. } => ParamValue::Path(None),
        }
    }
}
