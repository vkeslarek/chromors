use std::sync::Arc;

/// Creates an aligned staging buffer, row-by-row copy from tight source
/// at (src_x, src_y) within the source buffer. Returns (aligned_buffer, aligned_bytes_per_row).
pub(crate) fn stage_aligned(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    src_row_bytes: u32,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
    bpp: u32,
) -> (Arc<wgpu::Buffer>, u32) {
    let _span = tracing::trace_span!("tile.stage").entered();
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let align_mask = !(align - 1);
    let row_align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let raw_row = (width * bpp).max(width * 4);
    let aligned_bpr = if !raw_row.is_multiple_of(row_align) {
        raw_row + row_align - (raw_row % row_align)
    } else {
        raw_row
    };

    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp_fetch_stage"),
        size: (aligned_bpr * height) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let src_byte_offset = (src_x * bpp) as u64;
    let aligned_src_off = src_byte_offset & align_mask;
    let raw_end = src_byte_offset + (width * bpp) as u64;
    let aligned_end = (raw_end + align - 1) & align_mask;
    let max_end = src_row_bytes as u64;
    let copy_end = aligned_end.min(max_end);
    let copy_bytes = copy_end - aligned_src_off;

    for row in 0..height {
        let src_off = ((src_y + row) as u64 * src_row_bytes as u64) + aligned_src_off;
        let dst_off = row as u64 * aligned_bpr as u64;
        enc.copy_buffer_to_buffer(src, src_off, &stage, dst_off, copy_bytes);
    }
    queue.submit(std::iter::once(enc.finish()));
    (Arc::new(stage), aligned_bpr)
}
