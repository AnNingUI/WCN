#ifndef WCN_INSTANCE_EXPANDER_WGSL_H
#define WCN_INSTANCE_EXPANDER_WGSL_H

#include "WCN/WCN_WGSL.h"

static const char* WCN_INSTANCE_EXPANDER_WGSL = WGSL_CODE(
const INSTANCE_TYPE_RECT: u32 = 0u;
const INSTANCE_TYPE_TEXT: u32 = 1u;
const INSTANCE_TYPE_PATH: u32 = 2u;
const INSTANCE_TYPE_LINE: u32 = 3u;

// Line flag bits (shared with render shader)
const LINE_CAP_BUTT: u32 = 0u;
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

fn unpack_color(packed: u32) -> vec4<f32> {
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn build_vertex(
    instance: Instance,
    local_pos: vec2<f32>,
    sized_pos: vec2<f32>
) -> VertexData {
    let m00 = instance.transform.x;
    let m01 = instance.transform.y;
    let m10 = instance.transform.z;
    let m11 = instance.transform.w;

    let transformed_x = sized_pos.x * m00 + sized_pos.y * m01;
    let transformed_y = sized_pos.x * m10 + sized_pos.y * m11;
    let world_pos = vec2<f32>(transformed_x, transformed_y) + instance.position;

    let ndc_x = (world_pos.x / uniforms.viewport_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world_pos.y / uniforms.viewport_size.y) * 2.0;

    var vertex: VertexData;
    vertex.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    vertex.color = unpack_color(instance.color);
    vertex.uv = instance.uv + local_pos * instance.uvSize;
    vertex.instance_type = instance.instance_type;
    vertex.flags = instance.flags;
    vertex.local_pos = local_pos;
    vertex.params_x = instance.params_x;
    vertex.padding0 = 0.0;
    vertex.size = instance.size;
    vertex.tri_v0 = vec2<f32>(0.0, 0.0);
    vertex.tri_v1 = vec2<f32>(0.0, 0.0);
    vertex.tri_v2 = vec2<f32>(0.0, 0.0);
    if (instance.instance_type == INSTANCE_TYPE_PATH) {
        let safe_size = vec2<f32>(
            max(instance.size.x, 1e-4),
            max(instance.size.y, 1e-4)
        );
        let tri0 = (vec2<f32>(instance.uv.x, instance.uv.y) - instance.position) / safe_size;
        let tri1 = (vec2<f32>(instance.uvSize.x, instance.uvSize.y) - instance.position) / safe_size;
        let tri2 = (vec2<f32>(instance.params_x, bitcast<f32>(instance.flags)) - instance.position) / safe_size;
        vertex.tri_v0 = tri0;
        vertex.tri_v1 = tri1;
        vertex.tri_v2 = tri2;
    }
    return vertex;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_index = global_id.x;
    if (batch_index >= uniforms.instance_count) {
        return;
    }

    let instance_index = uniforms.instance_offset + batch_index;
    let instance = instances[instance_index];
    let local_positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );

    var vertex_base = batch_index * 6u;

    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let local_pos = local_positions[i];
        var sized_pos = local_pos * instance.size;

        if (instance.instance_type == INSTANCE_TYPE_LINE) {
            let dir = instance.uv;
            let perp = vec2<f32>(-dir.y, dir.x);
            let length = instance.size.x;
            let width = instance.size.y;
            let half_width = width * 0.5;

            let start_cap = (instance.flags & LINE_CAP_START_ENABLED) != 0u;
            let end_cap = (instance.flags & LINE_CAP_END_ENABLED) != 0u;
            var start_extension = 0.0;
            var end_extension = 0.0;
            if (start_cap) {
                start_extension = half_width;
            }
            if (end_cap) {
                end_extension = half_width;
            }
            let extended_length = length + start_extension + end_extension;
            let center_offset = (end_extension - start_extension) * 0.5;
            let along = (local_pos.x - 0.5) * extended_length + center_offset;
            let across = (local_pos.y - 0.5) * width;
            sized_pos = vec2<f32>(
                along * dir.x + across * perp.x,
                along * dir.y + across * perp.y
            );
        }

        let vertex = build_vertex(instance, local_pos, sized_pos);
        vertices[vertex_base + i] = vertex;
    }
}
);

#endif // WCN_INSTANCE_EXPANDER_WGSL_H
