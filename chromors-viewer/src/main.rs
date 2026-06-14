//! Windowed image viewer on top of `chromors-viewport`.

mod app;
mod gpu;
mod editor;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("poc=debug".parse().unwrap())
                .add_directive("app=info".parse().unwrap())
                .add_directive("warn".parse().unwrap()),
        )
        .with_thread_names(true)
        .with_target(false)
        .init();

    // Optional positional arg: an image path to open on startup (skips Ctrl+O).
    let initial = std::env::args().nth(1).map(std::path::PathBuf::from);
    app::ImageViewerApp::new().run(initial);
}
