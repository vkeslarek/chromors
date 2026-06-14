// lib/color/interp.slang — color reinterpretation as view wraps (§5.12).
//
// The first instance of the generic read/write wrap framework (§5.5-§5.10,
// lib/region.slang's `IRegion`/`IWritableRegion`): nests `color_convert`
// (lib/color/convert.slang) around any region, on read or on write. Built by
// `color_read_wrap`/`color_write_wrap` (`src/backend/gpu/color_params.rs`)
// and spliced in by `GpuBuilder::read_wrap`/`write_wrap`.

import lib.color.convert;
import lib.color.params;
import lib.region;

/// Read: reinterpret a region from `(model,space,alpha)_src` to `_dst` via
/// the precomputed `p`. Generic over ANY `IRegion` — the wrapped source/temp
/// is decoded as usual, then every sample runs through `color_convert`.
struct ColorReadView<R : IRegion> : IRegion {
    R inner;
    ColorConvertParams p;

    float4 read(uint2 idx) {
        float4 c = inner.read(idx);
        return color_convert(c, c.a, p);
    }

    float4 read_clamped(int2 idx) {
        float4 c = inner.read_clamped(idx);
        return color_convert(c, c.a, p);
    }
}

/// Write: convert a working `float4` to the destination interpretation, then
/// delegate to the wrapped sink (codec/temp). Generic over ANY
/// `IWritableRegion`.
struct ColorWriteSink<W : IWritableRegion> : IWritableRegion {
    W inner;
    ColorConvertParams p;

    void write(uint2 idx, float4 value) {
        inner.write(idx, color_convert(value, value.a, p));
    }
}
