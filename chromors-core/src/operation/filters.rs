use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Sharpen ───────────────────────────────────────────────────────────────────

pub struct Sharpen<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub sigma: Option<f64>,
    pub flat: Option<f64>,
    pub jagged: Option<f64>,
    pub edge: Option<f64>,
    pub smooth: Option<f64>,
    pub maximum: Option<f64>,
}

impl<B: Backend> Operation<B> for Sharpen<B>
where
    Sharpen<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = self.sigma.unwrap_or(1.0) * 3.0; // typical 3-sigma
        vec![Some(WorkUnit::Region(out.expanded(halo.ceil() as i32)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.sigma.unwrap_or(0.0).to_bits());
        state.write_u64(self.flat.unwrap_or(0.0).to_bits());
        state.write_u64(self.jagged.unwrap_or(0.0).to_bits());
        state.write_u64(self.edge.unwrap_or(0.0).to_bits());
        state.write_u64(self.smooth.unwrap_or(0.0).to_bits());
        state.write_u64(self.maximum.unwrap_or(0.0).to_bits());
    }
}

// ── Canny ─────────────────────────────────────────────────────────────────────

pub struct Canny<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub sigma: Option<f64>,
    pub precision: Option<i32>,
}

impl<B: Backend> Operation<B> for Canny<B>
where
    Canny<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let sigma = self.sigma.unwrap_or(1.4) as f32 / out.lod.scale_factor() as f32;
        // gauss_radius for the blur, +1 for the 2x2 gradient corner, +1 for
        // the directional neighbor read in non-max suppression.
        let halo = gauss_radius(sigma) + 2;
        vec![Some(WorkUnit::Region(out.expanded(halo)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.sigma.unwrap_or(0.0).to_bits());
        state.write_i32(self.precision.unwrap_or(0));
    }
}

// ── Median ────────────────────────────────────────────────────────────────────

pub struct Median<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub size: i32,
}

impl<B: Backend> Operation<B> for Median<B>
where
    Median<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = self.size / 2;
        vec![Some(WorkUnit::Region(out.expanded(halo)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.size);
    }
}

// ── HoughLine ─────────────────────────────────────────────────────────────────

pub struct HoughLine<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: Option<i32>,
    pub height: Option<i32>,
}

impl<B: Backend> Operation<B> for HoughLine<B>
where
    HoughLine<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // Hough usually needs full image, but for now we demand the same region
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width.unwrap_or(0));
        state.write_i32(self.height.unwrap_or(0));
    }
}

// ── HoughCircle ───────────────────────────────────────────────────────────────

pub struct HoughCircle<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub scale: Option<i32>,
    pub min_radius: Option<i32>,
    pub max_radius: Option<i32>,
}

impl<B: Backend> Operation<B> for HoughCircle<B>
where
    HoughCircle<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // Hough usually needs full image, but for now we demand the same region
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.scale.unwrap_or(0));
        state.write_i32(self.min_radius.unwrap_or(0));
        state.write_i32(self.max_radius.unwrap_or(0));
    }
}

// ── Blur ──────────────────────────────────────────────────────────────────────

pub struct Blur<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub sigma: f32,
}

impl<B: Backend> Operation<B> for Blur<B>
where
    Blur<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = gauss_radius(self.sigma / out.lod.scale_factor() as f32);
        vec![Some(WorkUnit::Region(out.expanded(halo)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.sigma.to_bits());
    }
}

/// Mask radius matching `vips_gaussmat`'s default (`min_ampl = 0.2`): the
/// largest `x` with `exp(-x^2 / (2*sigma^2)) >= min_ampl`. Much narrower than
/// a naive `sigma*3` window -- e.g. radius 5 (not 9) for sigma=3.
pub fn gauss_radius(sigma: f32) -> i32 {
    let sig2 = 2.0 * sigma * sigma;
    let max_x = (8.0 * sigma) as i32;
    let mut x = 0;
    while x < max_x {
        let v = (-((x * x) as f32) / sig2).exp();
        if v < 0.2 {
            break;
        }
        x += 1;
    }
    (x - 1).max(0)
}

impl<B: Backend> crate::data::image::Image2D<B>
where
    Blur<B>: Lower<B>,
{
    pub fn blur(&self, sigma: f32) -> Self {
        self.push(Blur {
            input: self.as_input(),
            sigma,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Sharpen<B>: crate::operation::Lower<B>,
{
    pub fn sharpen(
        &self,
        sigma: Option<f64>,
        flat: Option<f64>,
        jagged: Option<f64>,
        edge: Option<f64>,
        smooth: Option<f64>,
        maximum: Option<f64>,
    ) -> Self {
        self.push(Sharpen {
            input: self.as_input(),
            sigma,
            flat,
            jagged,
            edge,
            smooth,
            maximum,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Canny<B>: crate::operation::Lower<B>,
{
    pub fn canny(&self, sigma: Option<f64>, precision: Option<i32>) -> Self {
        self.push(Canny {
            input: self.as_input(),
            sigma,
            precision,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Median<B>: crate::operation::Lower<B>,
{
    pub fn median(&self, size: i32) -> Self {
        self.push(Median {
            input: self.as_input(),
            size,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HoughLine<B>: crate::operation::Lower<B>,
{
    pub fn hough_line(&self, width: Option<i32>, height: Option<i32>) -> Self {
        self.push(HoughLine {
            input: self.as_input(),
            width,
            height,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HoughCircle<B>: crate::operation::Lower<B>,
{
    pub fn hough_circle(
        &self,
        scale: Option<i32>,
        min_radius: Option<i32>,
        max_radius: Option<i32>,
    ) -> Self {
        self.push(HoughCircle {
            input: self.as_input(),
            scale,
            min_radius,
            max_radius,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Blur<B>: crate::operation::Lower<B>,
    crate::operation::arithmetic::Multiply<B>: crate::operation::Lower<B>,
    crate::operation::arithmetic::Subtract<B>: crate::operation::Lower<B>,
    crate::operation::arithmetic::Add<B>: crate::operation::Lower<B>,
    crate::operation::arithmetic::Divide<B>: crate::operation::Lower<B>,
    crate::operation::arithmetic::Linear<B>: crate::operation::Lower<B>,
{
    pub fn guided_filter(&self, p: &crate::data::image::Image2D<B>, radius: f32, eps: f64) -> Self {
        // Guidance image: self (I)
        // Filtering input image: p
        
        let mean_i = self.blur(radius);
        let mean_p = p.blur(radius);

        let ii = self.multiply(self);
        let mean_ii = ii.blur(radius);

        let ip = self.multiply(p);
        let mean_ip = ip.blur(radius);

        let mean_i_mean_i = mean_i.multiply(&mean_i);
        let var_i = mean_ii.subtract(&mean_i_mean_i);

        let mean_i_mean_p = mean_i.multiply(&mean_p);
        let cov_ip = mean_ip.subtract(&mean_i_mean_p);

        // var_i + eps
        let var_i_eps = var_i.linear(vec![1.0], vec![eps]);

        // a = cov_ip / (var_i + eps)
        let a = cov_ip.divide(&var_i_eps);
        
        // b = mean_p - a * mean_i
        let a_mean_i = a.multiply(&mean_i);
        let b = mean_p.subtract(&a_mean_i);

        let mean_a = a.blur(radius);
        let mean_b = b.blur(radius);

        // q = mean_a * I + mean_b
        let mean_a_i = mean_a.multiply(self);
        mean_a_i.add(&mean_b)
    }
}
