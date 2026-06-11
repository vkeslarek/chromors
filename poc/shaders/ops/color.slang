import lib.region;
import lib.color.convert;
import lib.color.params;

public void color_convert_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, ColorConvertParams p)
{
    // Note: The POC doesn't support complex structs in params yet, 
    // but this port proves the color conversion shader is adapted to the new IRegion architecture.
    float4 c = input.read(idx);
    float4 converted = color_convert(c, 1.0, p);
    output.write(idx, converted);
}
