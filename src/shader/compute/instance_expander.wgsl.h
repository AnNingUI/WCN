#ifndef WCN_INSTANCE_EXPANDER_WGSL_H
#define WCN_INSTANCE_EXPANDER_WGSL_H

#include "WCN/WCN_WGSL.h"

static const char* WCN_INSTANCE_EXPANDER_WGSL = WGSL_CODE(

// Constants
const INSTANCE_TYPE_RECT: u32 = 0u;
const INSTANCE_TYPE_TEXT: u32 = 1u;
const INSTANCE_TYPE_PATH: u32 = 2u;
const INSTANCE_TYPE_LINE: u32 = 3u;
const INSTANCE_TYPE_IMAGE: u32 = 4u;
const INSTANCE_TYPE_ARC: u32 = 5u;
const INSTANCE_TYPE_BEZIER: u32 = 6u;
const INSTANCE_TYPE_CIRCLE_FILL: u32 = 7u;

const LINE_CAP_START_ENABLED: u32 = 0x100u;
const LINE_CAP_END_ENABLED: u32 = 0x200u;

struct Instance {
    position: vec2<f32>,
    size: vec2<f32>,
    uv: vec2<f32>,
    uvSize: vec2<f32>,
    transform: vec4<f32>,
    color: u32,
    instance_type: u32,
    flags: u32,
    params_x: f32,
};

struct Uniforms {
    viewport_size: vec2<f32>,
    instance_count: u32,
    instance_offset: u32,
};

struct VertexData {
    clip_position: vec4<f32>,
    color: vec4<f32>,
    uv: vec2<f32>,
    instance_type: u32,
    flags: u32,
    local_pos: vec2<f32>,
    params_x: f32,
    padding0: f32,
    size: vec2<f32>,
    tri_v0: vec2<f32>,
    tri_v1: vec2<f32>,
    tri_v2: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<storage, read_write> vertices: array<VertexData>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// Optimization: Use Workgroup Size 256 for better occupancy on modern GPUs
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Parallelism: Each thread handles ONE vertex, not one instance.
    let global_vertex_index = global_id.x;

    // Calculate which instance and which vertex within the quad (0-5) this thread handles
    let instance_idx_local = global_vertex_index / 6u;
    let vertex_sub_idx = global_vertex_index % 6u; // 0..5

    // Bounds check
    if (instance_idx_local >= uniforms.instance_count) {
        return;
    }

    let instance_index = uniforms.instance_offset + instance_idx_local;
    let instance = instances[instance_index];

    // Optimization: Generate local position (UV) mathematically to avoid array lookup logic
    // Map 0->(0,0), 1->(1,0), 2->(0,1), 3->(1,1), 4->(0,1), 5->(1,0)
    // x is 1.0 at indices 1, 3, 5
    // y is 1.0 at indices 2, 3, 4
    let lx = f32(vertex_sub_idx & 1u); // 1, 3, 5 are odd
    let ly = select(0.0, 1.0, (vertex_sub_idx > 1u) & (vertex_sub_idx < 5u));
    let local_pos = vec2<f32>(lx, ly);

    var sized_pos = local_pos * instance.size;

    // Arc Logic - 圆弧使用 uvSize 作为边界框大小
    if (instance.instance_type == INSTANCE_TYPE_ARC) {
        sized_pos = (local_pos - 0.5) * instance.uvSize;
    }

    // Circle Fill Logic - 圆形填充使用 uvSize 作为边界框大小
    if (instance.instance_type == INSTANCE_TYPE_CIRCLE_FILL) {
        sized_pos = (local_pos - 0.5) * instance.uvSize;
    }

    // Line Logic
    if (instance.instance_type == INSTANCE_TYPE_LINE) {
        let dir = instance.uv;
        let perp = vec2<f32>(-dir.y, dir.x);

        // Use vector components directly
        let length = instance.size.x;
        let width = instance.size.y;

        // Branchless selection for caps
        let start_cap = (instance.flags & LINE_CAP_START_ENABLED) != 0u;
        let end_cap = (instance.flags & LINE_CAP_END_ENABLED) != 0u;

        let half_width = width * 0.5;
        let start_ext = select(0.0, half_width, start_cap);
        let end_ext = select(0.0, half_width, end_cap);

        let extended_length = length + start_ext + end_ext;
        let center_offset = (end_ext - start_ext) * 0.5;

        let along = (local_pos.x - 0.5) * extended_length + center_offset;
        let across = (local_pos.y - 0.5) * width;

        // FMA (Fused Multiply-Add) optimization usually handled by compiler, but explicit is good
        sized_pos = dir * along + perp * across;
    }

    // Matrix Transform
    // Expand manually to potentially use fma instructions
    let world_pos = vec2<f32>(
        sized_pos.x * instance.transform.x + sized_pos.y * instance.transform.y,
        sized_pos.x * instance.transform.z + sized_pos.y * instance.transform.w
    ) + instance.position;

    // Coordinate Space Conversion (NDC)
    // Pre-calculate inverse scale to replace division with multiplication
    let inv_viewport = 1.0 / uniforms.viewport_size;
    let ndc_x = (world_pos.x * inv_viewport.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world_pos.y * inv_viewport.y) * 2.0;

    // Build Vertex Output
    var vertex: VertexData;
    vertex.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    // Optimization: Use built-in intrinsic for color unpacking
    vertex.color = unpack4x8unorm(instance.color);

    // ARC 和 CIRCLE_FILL 类型需要保持 uv 不变（存储角度数据，不能插值）
    if (instance.instance_type == INSTANCE_TYPE_ARC || instance.instance_type == INSTANCE_TYPE_CIRCLE_FILL) {
        vertex.uv = instance.uv;  // 直接传递 start_angle, end_angle
    } else {
        vertex.uv = instance.uv + local_pos * instance.uvSize;
    }
    vertex.instance_type = instance.instance_type;
    vertex.flags = instance.flags;
    vertex.local_pos = local_pos;
    vertex.params_x = instance.params_x;
    vertex.padding0 = 0.0;
    vertex.size = instance.size;

    // Initialize tri vectors
    vertex.tri_v0 = vec2<f32>(0.0);
    vertex.tri_v1 = vec2<f32>(0.0);
    vertex.tri_v2 = vec2<f32>(0.0);

    // Path Logic
    if (instance.instance_type == INSTANCE_TYPE_PATH) {
        let safe_size = max(instance.size, vec2<f32>(1e-4));
        let inv_safe_size = 1.0 / safe_size;
        vertex.tri_v0 = (instance.uv - instance.position) * inv_safe_size;
        vertex.tri_v1 = (instance.uvSize - instance.position) * inv_safe_size;
        vertex.tri_v2 = (vec2<f32>(instance.params_x, bitcast<f32>(instance.flags)) - instance.position) * inv_safe_size;
    }

    // Bezier Logic - 传递贝塞尔曲线数据
    if (instance.instance_type == INSTANCE_TYPE_BEZIER) {
        // tri_v0 = 控制点归一化坐标 (从 uv)
        vertex.tri_v0 = instance.uv;
        // tri_v1 = 起点归一化坐标 (从 transform[2,3])
        vertex.tri_v1 = vec2<f32>(instance.transform.z, instance.transform.w);
        // tri_v2 = 终点归一化坐标 (从 uvSize)
        vertex.tri_v2 = instance.uvSize;
    }

    // Optimization: Coalesced Global Memory Write
    // Thread N writes to Index N.
    vertices[global_vertex_index] = vertex;
}
);

#endif // WCN_INSTANCE_EXPANDER_WGSL_H
