// One oversized triangle that covers the whole viewport: no vertex buffer, no index buffer.
// index 0 -> (-1,-1) uv (0,0), index 1 -> (3,-1) uv (2,0), index 2 -> (-1,3) uv (0,2).

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let i = i32(vertex_index);
    let x = f32(i & 1) * 4.0 - 1.0;
    let y = f32(i >> 1) * 4.0 - 1.0;
    var out: VertexOut;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (y + 1.0) * 0.5);
    return out;
}
