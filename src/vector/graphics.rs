use vello::Scene;

/// O Trait que o motor vai entender para renderizar gráficos vetoriais.
pub trait VectorGraphics: Send {
    fn draw(&self, scene: &mut Scene);
}
