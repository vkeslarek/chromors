import lib.color.transfer;

struct Srgb {
    static float3 to_linear(float3 c)
    {
        return float3(
            decode_tf(c.r, TransferFn::Srgb),
            decode_tf(c.g, TransferFn::Srgb),
            decode_tf(c.b, TransferFn::Srgb)
        );
    }

    static float3 from_linear(float3 c)
    {
        return float3(
            encode_tf(c.r, TransferFn::Srgb),
            encode_tf(c.g, TransferFn::Srgb),
            encode_tf(c.b, TransferFn::Srgb)
        );
    }
}
