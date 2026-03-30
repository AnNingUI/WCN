#ifndef WCN_INSTANCE_EXPANDER_WGSL_H
#define WCN_INSTANCE_EXPANDER_WGSL_H

#include "WCN/WCN_WGSL.h"

static const char* WCN_INSTANCE_EXPANDER_WGSL = WGSL_CODE(

const INSTANCE_TYPE_RECT: u32 = 0u;
const INSTANCE_TYPE_TEXT: u32 = 1u;
const INSTANCE_TYPE_PATH: u32 = 2u;
const INSTANCE_TYPE_LINE: u32 = 3u;
const INSTANCE_TYPE_IMAGE: u32 = 4u;
const INSTANCE_TYPE_ARC: u32 = 5u;
const INSTANCE_TYPE_BEZIER: u32 = 6u;
const INSTANCE_TYPE_CIRCLE_FILL: u32 = 7u;

const CIRCLE_FILL_FLAG_RAW_JOIN_NORMALS: u32 = 0x40000000u;
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_vertex_index = global_id.x;
    let instance_idx_local = global_vertex_index / 6u;
    let vertex_sub_idx = global_vertex_index % 6u;

    if (instance_idx_local >= uniforms.instance_count) {
        return;
    }

    let instance_index = uniforms.instance_offset + instance_idx_local;
    let instance = instances[instance_index];

    let PI = 3.14159265;
    let TWO_PI = 6.283185307;

    let lx = f32(vertex_sub_idx & 1u);
    let ly = select(0.0, 1.0, (vertex_sub_idx > 1u) & (vertex_sub_idx < 5u));
    let local_pos = vec2<f32>(lx, ly);

    var sized_pos = local_pos * instance.size;
    var circle_start_angle = instance.uv.x;
    var circle_end_angle = instance.uv.y;

    if (instance.instance_type == INSTANCE_TYPE_ARC) {
        sized_pos = (local_pos - 0.5) * instance.uvSize;
    }

    if (instance.instance_type == INSTANCE_TYPE_CIRCLE_FILL) {
        if ((instance.flags & CIRCLE_FILL_FLAG_RAW_JOIN_NORMALS) != 0u) {
            let n1_len = max(length(instance.uv), 1e-6);
            let n2_len = max(length(instance.uvSize), 1e-6);
            let n1 = instance.uv / n1_len;
            let n2 = instance.uvSize / n2_len;
            let a1 = atan2(n1.y, n1.x);
            let a2 = atan2(n2.y, n2.x);
            var diff = a2 - a1;
            if (diff > PI) {
                diff = diff - TWO_PI;
            }
            if (diff < -PI) {
                diff = diff + TWO_PI;
            }
            circle_start_angle = a1;
            circle_end_angle = a1 + diff;
            sized_pos = (local_pos - 0.5) * vec2<f32>(instance.size.x * 2.0);
        } else {
            sized_pos = (local_pos - 0.5) * instance.uvSize;
        }
    }

    if (instance.instance_type == INSTANCE_TYPE_LINE) {
        let dir = instance.uv;
        let perp = vec2<f32>(-dir.y, dir.x);

        let length = instance.size.x;
        let width = instance.size.y;

        let start_cap = (instance.flags & LINE_CAP_START_ENABLED) != 0u;
        let end_cap = (instance.flags & LINE_CAP_END_ENABLED) != 0u;

        let half_width = width * 0.5;
        let start_ext = select(0.0, half_width, start_cap);
        let end_ext = select(0.0, half_width, end_cap);

        let extended_length = length + start_ext + end_ext;
        let center_offset = (end_ext - start_ext) * 0.5;

        let along = (local_pos.x - 0.5) * extended_length + center_offset;
        let across = (local_pos.y - 0.5) * width;

        sized_pos = dir * along + perp * across;
    }

    var world_pos: vec2<f32>;
    if (instance.instance_type == INSTANCE_TYPE_PATH) {
        world_pos = sized_pos + instance.position;
    } else {
        world_pos = vec2<f32>(
            sized_pos.x * instance.transform.x + sized_pos.y * instance.transform.y,
            sized_pos.x * instance.transform.z + sized_pos.y * instance.transform.w
        ) + instance.position;
    }

    let inv_viewport = 1.0 / uniforms.viewport_size;
    let ndc_x = (world_pos.x * inv_viewport.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world_pos.y * inv_viewport.y) * 2.0;

    var vertex: VertexData;
    vertex.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    vertex.color = unpack4x8unorm(instance.color);

    if (instance.instance_type == INSTANCE_TYPE_ARC) {
        vertex.uv = instance.uv;
    } else if (instance.instance_type == INSTANCE_TYPE_CIRCLE_FILL) {
        vertex.uv = vec2<f32>(circle_start_angle, circle_end_angle);
    } else {
        vertex.uv = instance.uv + local_pos * instance.uvSize;
    }

    vertex.instance_type = instance.instance_type;
    vertex.flags = instance.flags;
    vertex.local_pos = local_pos;
    vertex.params_x = instance.params_x;
    vertex.padding0 = 0.0;
    vertex.size = instance.size;

    vertex.tri_v0 = vec2<f32>(0.0);
    vertex.tri_v1 = vec2<f32>(0.0);
    vertex.tri_v2 = vec2<f32>(0.0);

    if (instance.instance_type == INSTANCE_TYPE_PATH) {
        let safe_size = max(instance.size, vec2<f32>(1e-4));
        let inv_safe_size = 1.0 / safe_size;
        vertex.tri_v0 = (instance.uv - instance.position) * inv_safe_size;
        vertex.tri_v1 = (instance.uvSize - instance.position) * inv_safe_size;
        vertex.tri_v2 = (vec2<f32>(instance.params_x, bitcast<f32>(instance.flags)) - instance.position) * inv_safe_size;
        vertex.params_x = instance.transform.z;
    }

    if (instance.instance_type == INSTANCE_TYPE_BEZIER) {
        vertex.tri_v0 = instance.uv;
        vertex.tri_v1 = vec2<f32>(instance.transform.z, instance.transform.w);
        vertex.tri_v2 = instance.uvSize;
    }

    let output_vertex_index = instance_index * 6u + vertex_sub_idx;
    vertices[output_vertex_index] = vertex;
}
);

#endif // WCN_INSTANCE_EXPANDER_WGSL_H
