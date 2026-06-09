// Shared blur parameters — used by GaussianBlurOp and SeparableBlurOp.
// Not an operation itself, just data layout.

public struct GaussianBlurParams
{
    public int   radius;
    public float sigma;
    public int   center_x;
    public int   center_y;
}
