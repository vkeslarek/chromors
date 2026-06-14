use parley::layout::{Alignment, Layout};
use parley::{FontContext, LayoutContext};
use std::sync::OnceLock;
use vello::Scene;
use vello::kurbo::{Affine, Point};
use vello::peniko::{Brush, Color};

pub fn font_context() -> &'static std::sync::Mutex<FontContext> {
    static FONT_CTX: OnceLock<std::sync::Mutex<FontContext>> = OnceLock::new();
    FONT_CTX.get_or_init(|| {
        let mut ctx = FontContext::default();
        let font_data = include_bytes!("../../assets/Inter-Regular.ttf");
        ctx.collection.register_fonts(font_data.to_vec());
        std::sync::Mutex::new(ctx)
    })
}

thread_local! {
    static LAYOUT_CTX: std::cell::RefCell<LayoutContext<Brush>> = std::cell::RefCell::new(LayoutContext::new());
}

fn build_layout(text: &str, size: f32, brush: Brush) -> Layout<Brush> {
    let mut fcx = font_context().lock().unwrap();
    LAYOUT_CTX.with(|lcx| {
        let mut lcx = lcx.borrow_mut();
        let mut builder = lcx.ranged_builder(&mut fcx, text, 1.0);
        builder.push_default(parley::style::StyleProperty::Brush(brush));
        builder.push_default(parley::style::StyleProperty::FontSize(size));
        let mut layout = builder.build(text);
        layout.break_all_lines(None);
        layout
    })
}

pub fn draw_line(scene: &mut Scene, transform: Affine, text: &str, size: f64, color: Color) {
    let brush = Brush::Solid(color);
    let layout = build_layout(text, size as f32, brush);

    for line in layout.lines() {
        for item in line.items() {
            if let parley::layout::PositionedLayoutItem::GlyphRun(glyph_run) = item {
                let run = glyph_run.run();
                let style = glyph_run.style();
                let parley_font = run.font();
                let blob = vello::peniko::Blob::new(std::sync::Arc::new(
                    parley_font.data.as_ref().to_vec(),
                ));
                let vello_font = vello::peniko::FontData::new(blob, parley_font.index);
                let run_transform = transform
                    * vello::kurbo::Affine::translate((
                        glyph_run.offset() as f64,
                        line.metrics().baseline as f64,
                    ));
                let mut cursor_x = 0.0f32;
                scene
                    .draw_glyphs(&vello_font)
                    .brush(&style.brush)
                    .hint(true)
                    .transform(run_transform)
                    .glyph_transform(None)
                    .font_size(run.font_size())
                    .draw(
                        vello::peniko::Fill::NonZero,
                        glyph_run.glyphs().map(|g| {
                            let gx = cursor_x + g.x;
                            cursor_x += g.advance;
                            vello::Glyph {
                                id: g.id as _,
                                x: gx,
                                y: g.y,
                            }
                        }),
                    );
            }
        }
    }
}

pub fn measure(text: &str, size: f64) -> f64 {
    let layout = build_layout(text, size as f32, Brush::Solid(Color::WHITE));
    layout.width() as f64
}
