#ifndef WCN_FULLSTACK_SHADERS_H
#define WCN_FULLSTACK_SHADERS_H

#include "WCN/WCN_WGSL.h"

static const char* FS_COMPUTE_WGSL = WGSL_CODE(
const CMD_RECT: u32 = 0u;
const CMD_IMAGE: u32 = 1u;
const CMD_TEXT: u32 = 2u;
const CMD_LINE: u32 = 3u;
const CMD_PATH_SEGMENT: u32 = 4u;
const CMD_CIRCLE: u32 = 5u;
const CMD_ARC: u32 = 6u;
const CMD_BEZIER_QUAD: u32 = 7u;
const CMD_RECT_STROKE: u32 = 8u;
const CMD_ELLIPSE: u32 = 9u;
const CMD_BEZIER_CUBIC: u32 = 10u;
const CMD_TRIANGLE: u32 = 11u;
const CLIP_RECT_BIT: u32 = 1u;
const CLIP_PATH_BIT: u32 = 2u;
const FS_RENDER_FLAG_ORIENTED_QUAD: u32 = 1u << 14u;
const FS_RENDER_FLAG_LOCAL_SPACE: u32 = 1u << 15u;
const FS_RENDER_FLAG_PATTERN_SHADE: u32 = 1u << 13u;
const FS_RENDER_FLAG_IMAGE_NEAREST: u32 = 1u << 1u;
const FS_RENDER_FLAG_CLIP_MASK: u32 = 1u << 16u;
const FS_RENDER_FLAG_CLIP_PARENT_SHIFT: u32 = 17u;
const FS_RENDER_FLAG_CLIP_PARENT_MASK: u32 = 0x7Fu << FS_RENDER_FLAG_CLIP_PARENT_SHIFT;
const FS_RENDER_FLAG_CLIP_LAYER_SHIFT: u32 = 24u;
const FS_RENDER_FLAG_CLIP_LAYER_MASK: u32 = 0xFFu << FS_RENDER_FLAG_CLIP_LAYER_SHIFT;
const FS_CLIP_LAYER_COUNT: u32 = 64u;
const FS_CLIP_LAYER_NONE: u32 = 0xFFFFFFFFu;

struct Command {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    quad0: vec4<f32>,
    quad1: vec4<f32>,
    clip_min: vec2<f32>,
    clip_max: vec2<f32>,
    clip_enabled: u32,
    state_index: u32,
    color_rgba8: u32,
    cmd_type: u32,
    flags: u32,
    scalar: f32,
};

struct CommandState {
    clip_rect: vec4<f32>,
    clip_meta: vec4<u32>,
    xform0: vec4<f32>,
    xform1: vec4<f32>,
    pattern_inv0: vec4<f32>,
    pattern_inv1: vec4<f32>,
    pattern_meta: vec4<f32>,
};

struct Uniforms {
    viewport: vec2<f32>,
    command_count: u32,
    clip_enabled: u32,
    clip_min: vec2<f32>,
    clip_max: vec2<f32>,
};

struct ClipLayers {
    parent: array<vec4<u32>, 16>,
    min_x: array<vec4<u32>, 16>,
    min_y: array<vec4<u32>, 16>,
    max_x: array<vec4<u32>, 16>,
    max_y: array<vec4<u32>, 16>,
};

struct VertexOut {
    clip_pos: vec4<f32>,
    color: vec4<f32>,
    uv: vec2<f32>,
    world_pos: vec2<f32>,
    cmd_type: u32,
    flags: u32,
    state_index: u32,
    _pad0: u32,
    extra0: vec4<f32>,
    extra1: vec4<f32>,
    extra2: vec4<f32>,
    extra3: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> commands: array<Command>;
@group(0) @binding(1) var<storage, read> command_states: array<CommandState>;
@group(0) @binding(2) var<storage, read_write> vertices: array<VertexOut>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<uniform> clip_layers: ClipLayers;

fn local_for_sub(sub: u32) -> vec2<f32> {
    switch (sub) {
        case 0u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u: { return vec2<f32>(0.0, 1.0); }
        case 3u: { return vec2<f32>(1.0, 0.0); }
        case 4u: { return vec2<f32>(1.0, 1.0); }
        default: { return vec2<f32>(0.0, 1.0); }
    }
}

fn clip_from_world(world: vec2<f32>, viewport: vec2<f32>) -> vec4<f32> {
    let ndc_x = (world.x / viewport.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world.y / viewport.y) * 2.0;
    return vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
}

fn lane_pick_u32(v: vec4<u32>, lane: u32) -> u32 {
    switch (lane) {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        case 2u: { return v.z; }
        default: { return v.w; }
    }
}

fn clip_parent_at(layer: u32) -> u32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return lane_pick_u32(clip_layers.parent[slot], lane);
}

fn clip_min_x_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.min_x[slot], lane));
}

fn clip_min_y_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.min_y[slot], lane));
}

fn clip_max_x_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.max_x[slot], lane));
}

fn clip_max_y_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.max_y[slot], lane));
}

fn oriented_uv_from_world(
    world: vec2<f32>,
    origin: vec2<f32>,
    du: vec2<f32>,
    dv: vec2<f32>,
    inv_det: f32
) -> vec2<f32> {
    let d = world - origin;
    let u = (d.x * dv.y - d.y * dv.x) * inv_det;
    let v = (d.y * du.x - d.x * du.y) * inv_det;
    return vec2<f32>(u, v);
}

fn transform_point(state: CommandState, p: vec2<f32>) -> vec2<f32> {
    let a = state.xform0.x;
    let b = state.xform0.y;
    let c = state.xform0.z;
    let d = state.xform0.w;
    let e = state.xform1.x;
    let f = state.xform1.y;
    return vec2<f32>(a * p.x + c * p.y + e, b * p.x + d * p.y + f);
}

fn transform_metric_scale(state: CommandState) -> f32 {
    let sx = length(vec2<f32>(state.xform0.x, state.xform0.y));
    let sy = length(vec2<f32>(state.xform0.z, state.xform0.w));
    return sqrt(max(sx * sy, 1e-8));
}

fn transform_length_x(state: CommandState, len: f32) -> f32 {
    let vx = vec2<f32>(state.xform0.x * len, state.xform0.y * len);
    return length(vx);
}

fn transform_length_y(state: CommandState, len: f32) -> f32 {
    let vy = vec2<f32>(state.xform0.z * len, state.xform0.w * len);
    return length(vy);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex_index = gid.x;
    let cmd_index = vertex_index / 6u;
    let sub = vertex_index % 6u;
    if (cmd_index >= uniforms.command_count) {
        return;
    }

    let cmd = commands[cmd_index];
    let state = command_states[cmd.state_index];
    let local_uv = local_for_sub(sub);
    let oriented_quad = (cmd.flags & FS_RENDER_FLAG_ORIENTED_QUAD) != 0u;
    let local_space = (cmd.flags & FS_RENDER_FLAG_LOCAL_SPACE) != 0u;

    var min_xy = vec2<f32>(0.0, 0.0);
    var max_xy = vec2<f32>(1.0, 1.0);
    var extra0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var extra1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var extra2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var extra3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var use_oriented_quad = false;
    var quad_origin = vec2<f32>(0.0, 0.0);
    var quad_du = vec2<f32>(0.0, 0.0);
    var quad_dv = vec2<f32>(0.0, 0.0);
    var quad_uv_min = vec2<f32>(0.0, 0.0);
    var quad_uv_max = vec2<f32>(1.0, 1.0);

    switch (cmd.cmd_type) {
        case CMD_RECT: {
            if (oriented_quad) {
                extra0 = vec4<f32>(cmd.p0.xy + cmd.p0.zw * 0.5, cmd.p0.zw);
                extra1 = cmd.p1;
                extra2 = cmd.p2;
                use_oriented_quad = true;
                quad_origin = cmd.quad0.xy;
                quad_du = cmd.quad0.zw;
                quad_dv = cmd.quad1.xy;
                let q1 = quad_origin + quad_du;
                let q2 = quad_origin + quad_dv;
                let q3 = quad_origin + quad_du + quad_dv;
                min_xy = min(min(quad_origin, q1), min(q2, q3));
                max_xy = max(max(quad_origin, q1), max(q2, q3));
            } else {
                let p00 = transform_point(state, cmd.p0.xy);
                let p11 = transform_point(state, cmd.p0.xy + cmd.p0.zw);
                let shape_min = min(p00, p11);
                let shape_max = max(p00, p11);
                let shape_size = shape_max - shape_min;
                let s = transform_metric_scale(state);
                let pad = 1.5;
                min_xy = shape_min - vec2<f32>(pad, pad);
                max_xy = shape_max + vec2<f32>(pad, pad);
                extra0 = vec4<f32>((shape_min + shape_max) * 0.5, shape_size);
                extra1 = cmd.p1 * s;
                extra2 = cmd.p2 * s;
            }
        }
        case CMD_RECT_STROKE: {
            if (oriented_quad) {
                extra0 = vec4<f32>(cmd.p0.xy + cmd.p0.zw * 0.5, cmd.p0.zw);
                extra1 = cmd.p1;
                extra2 = cmd.p2;
                extra3.x = cmd.scalar;
                use_oriented_quad = true;
                quad_origin = cmd.quad0.xy;
                quad_du = cmd.quad0.zw;
                quad_dv = cmd.quad1.xy;
                let q1 = quad_origin + quad_du;
                let q2 = quad_origin + quad_dv;
                let q3 = quad_origin + quad_du + quad_dv;
                min_xy = min(min(quad_origin, q1), min(q2, q3));
                max_xy = max(max(quad_origin, q1), max(q2, q3));
            } else {
                let p00 = transform_point(state, cmd.p0.xy);
                let p11 = transform_point(state, cmd.p0.xy + cmd.p0.zw);
                let shape_min = min(p00, p11);
                let shape_max = max(p00, p11);
                let shape_size = shape_max - shape_min;
                let s = transform_metric_scale(state);
                let stroke_w = cmd.scalar * s;
                let pad = stroke_w * 0.5 + 1.5;
                min_xy = shape_min - vec2<f32>(pad, pad);
                max_xy = shape_max + vec2<f32>(pad, pad);
                extra0 = vec4<f32>((shape_min + shape_max) * 0.5, shape_size);
                extra1 = cmd.p1 * s;
                extra2 = cmd.p2 * s;
                extra3.x = stroke_w;
            }
        }
        case CMD_IMAGE: {
            extra1 = cmd.p1;
            extra2 = cmd.p2;
            if (oriented_quad) {
                extra0 = vec4<f32>(cmd.p0.xy, cmd.p0.zw);
                use_oriented_quad = true;
                quad_origin = cmd.quad0.xy;
                quad_du = cmd.quad0.zw;
                quad_dv = cmd.quad1.xy;
                let q1 = quad_origin + quad_du;
                let q2 = quad_origin + quad_dv;
                let q3 = quad_origin + quad_du + quad_dv;
                min_xy = min(min(quad_origin, q1), min(q2, q3));
                max_xy = max(max(quad_origin, q1), max(q2, q3));
            } else {
                if (local_space) {
                    let p00 = transform_point(state, cmd.p0.xy);
                    let p10 = transform_point(state, cmd.p0.xy + vec2<f32>(cmd.p0.z, 0.0));
                    let p01 = transform_point(state, cmd.p0.xy + vec2<f32>(0.0, cmd.p0.w));
                    let p11 = transform_point(state, cmd.p0.xy + cmd.p0.zw);
                    min_xy = min(min(p00, p10), min(p01, p11));
                    max_xy = max(max(p00, p10), max(p01, p11));
                } else {
                    min_xy = cmd.p0.xy;
                    max_xy = cmd.p0.xy + cmd.p0.zw;
                }
                extra0 = vec4<f32>(min_xy, max_xy - min_xy);
            }
        }
        case CMD_TEXT: {
            extra1 = cmd.p1;
            extra2 = cmd.p2;
            if (local_space) {
                // Keep CPU text command payload in local units; resolve transform-dependent
                // stroke thickness in compute so path scaling stays in one stage.
                extra2.w = cmd.p2.w * transform_metric_scale(state);
            }
            if (oriented_quad) {
                extra0 = vec4<f32>(cmd.p0.xy, cmd.p0.zw);
                use_oriented_quad = true;
                quad_origin = cmd.quad0.xy;
                quad_du = cmd.quad0.zw;
                quad_dv = cmd.quad1.xy;
                let q1 = quad_origin + quad_du;
                let q2 = quad_origin + quad_dv;
                let q3 = quad_origin + quad_du + quad_dv;
                min_xy = min(min(quad_origin, q1), min(q2, q3));
                max_xy = max(max(quad_origin, q1), max(q2, q3));
            } else {
                if (local_space) {
                    let p00 = transform_point(state, cmd.p0.xy);
                    let p10 = transform_point(state, cmd.p0.xy + vec2<f32>(cmd.p0.z, 0.0));
                    let p01 = transform_point(state, cmd.p0.xy + vec2<f32>(0.0, cmd.p0.w));
                    let p11 = transform_point(state, cmd.p0.xy + cmd.p0.zw);
                    min_xy = min(min(p00, p10), min(p01, p11));
                    max_xy = max(max(p00, p10), max(p01, p11));
                } else {
                    min_xy = cmd.p0.xy;
                    max_xy = cmd.p0.xy + cmd.p0.zw;
                }
                extra0 = vec4<f32>(min_xy, max_xy - min_xy);
            }
        }
        case CMD_LINE, CMD_PATH_SEGMENT: {
            var a = cmd.p0.xy;
            var b = cmd.p0.zw;
            var width = cmd.scalar;
            if (local_space) {
                a = transform_point(state, cmd.p0.xy);
                b = transform_point(state, cmd.p0.zw);
                width = cmd.scalar * transform_metric_scale(state);
            }
            let half_w = max(width * 0.5, 1.0);
            min_xy = min(a, b) - vec2<f32>(half_w, half_w);
            max_xy = max(a, b) + vec2<f32>(half_w, half_w);
            extra0 = vec4<f32>(a, b);
            extra1 = vec4<f32>(width, 0.0, 0.0, 0.0);
        }
        case CMD_CIRCLE: {
            var c = cmd.p0.xy;
            var r = cmd.scalar;
            if (local_space) {
                c = transform_point(state, cmd.p0.xy);
                r = cmd.scalar * transform_metric_scale(state);
            }
            r = max(r, 1.0);
            min_xy = c - vec2<f32>(r, r);
            max_xy = c + vec2<f32>(r, r);
            extra0 = vec4<f32>(c, r, 0.0);
        }
        case CMD_ELLIPSE: {
            var c = cmd.p0.xy;
            var rx = cmd.p0.z;
            var ry = cmd.p0.w;
            if (local_space) {
                c = transform_point(state, cmd.p0.xy);
                rx = transform_length_x(state, cmd.p0.z);
                ry = transform_length_y(state, cmd.p0.w);
            }
            rx = max(rx, 1.0);
            ry = max(ry, 1.0);
            min_xy = c - vec2<f32>(rx, ry);
            max_xy = c + vec2<f32>(rx, ry);
            extra0 = vec4<f32>(c, rx, ry);
        }
        case CMD_ARC: {
            var c = cmd.p0.xy;
            var r = cmd.scalar;
            var t = cmd.p1.w;
            if (local_space) {
                c = transform_point(state, cmd.p0.xy);
                let s = transform_metric_scale(state);
                r = cmd.scalar * s;
                t = cmd.p1.w * s;
            }
            r = max(r, 1.0);
            t = max(t, 1.0);
            let outer = r + t * 0.5;
            min_xy = c - vec2<f32>(outer, outer);
            max_xy = c + vec2<f32>(outer, outer);
            extra0 = vec4<f32>(c, r, t);
            extra1 = vec4<f32>(cmd.p1.x, cmd.p1.y, 0.0, 0.0);
        }
        case CMD_BEZIER_QUAD: {
            var p0 = cmd.p0.xy;
            var cp = cmd.p0.zw;
            var p1 = cmd.p1.xy;
            var width = cmd.scalar;
            if (local_space) {
                p0 = transform_point(state, cmd.p0.xy);
                cp = transform_point(state, cmd.p0.zw);
                p1 = transform_point(state, cmd.p1.xy);
                width = cmd.scalar * transform_metric_scale(state);
            }
            let half_w = max(width * 0.5, 1.0);
            min_xy = min(min(p0, cp), p1) - vec2<f32>(half_w, half_w);
            max_xy = max(max(p0, cp), p1) + vec2<f32>(half_w, half_w);
            extra0 = vec4<f32>(p0, cp);
            extra1 = vec4<f32>(p1, width, 0.0);
        }
        case CMD_BEZIER_CUBIC: {
            var p0 = cmd.p0.xy;
            var c0 = cmd.p0.zw;
            var c1 = cmd.p1.xy;
            var p1 = cmd.p1.zw;
            var width = cmd.scalar;
            if (local_space) {
                p0 = transform_point(state, cmd.p0.xy);
                c0 = transform_point(state, cmd.p0.zw);
                c1 = transform_point(state, cmd.p1.xy);
                p1 = transform_point(state, cmd.p1.zw);
                width = cmd.scalar * transform_metric_scale(state);
            }
            let half_w = max(width * 0.5, 1.0);
            min_xy = min(min(p0, c0), min(c1, p1)) - vec2<f32>(half_w, half_w);
            max_xy = max(max(p0, c0), max(c1, p1)) + vec2<f32>(half_w, half_w);
            extra0 = vec4<f32>(p0, c0);
            extra1 = vec4<f32>(c1, p1);
            extra2 = vec4<f32>(width, 0.0, 0.0, 0.0);
        }
        case CMD_TRIANGLE: {
            var a = cmd.p0.xy;
            var b = cmd.p0.zw;
            var c = cmd.p1.xy;
            if (local_space) {
                a = transform_point(state, cmd.p0.xy);
                b = transform_point(state, cmd.p0.zw);
                c = transform_point(state, cmd.p1.xy);
            }
            min_xy = min(min(a, b), c);
            max_xy = max(max(a, b), c);
            extra0 = vec4<f32>(a, b);
            extra1 = vec4<f32>(c, cmd.p1.z, cmd.p1.w);
        }
        default: {
            min_xy = cmd.p0.xy;
            max_xy = cmd.p0.xy + vec2<f32>(1.0, 1.0);
        }
    }

    if ((state.clip_meta.x & CLIP_RECT_BIT) != 0u) {
        min_xy = max(min_xy, state.clip_rect.xy);
        max_xy = min(max_xy, state.clip_rect.zw);
        if (max_xy.x <= min_xy.x || max_xy.y <= min_xy.y) {
            var clipped_v: VertexOut;
            clipped_v.clip_pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
            clipped_v.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            clipped_v.uv = local_uv;
            clipped_v.world_pos = min_xy;
            clipped_v.cmd_type = cmd.cmd_type;
            clipped_v.flags = cmd.flags;
            clipped_v.state_index = cmd.state_index;
            clipped_v._pad0 = 0u;
            clipped_v.extra0 = extra0;
            clipped_v.extra1 = extra1;
            clipped_v.extra2 = extra2;
            clipped_v.extra3 = extra3;
            vertices[vertex_index] = clipped_v;
            return;
        }
    }

    if ((state.clip_meta.x & CLIP_PATH_BIT) != 0u) {
        var layer = state.clip_meta.y;
        var parent = state.clip_meta.z;
        var depth: u32 = 0u;
        loop {
            if (layer >= FS_CLIP_LAYER_COUNT) {
                var clipped_v: VertexOut;
                clipped_v.clip_pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
                clipped_v.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                clipped_v.uv = local_uv;
                clipped_v.world_pos = min_xy;
                clipped_v.cmd_type = cmd.cmd_type;
                clipped_v.flags = cmd.flags;
                clipped_v.state_index = cmd.state_index;
                clipped_v._pad0 = 0u;
                clipped_v.extra0 = extra0;
                clipped_v.extra1 = extra1;
                clipped_v.extra2 = extra2;
                clipped_v.extra3 = extra3;
                vertices[vertex_index] = clipped_v;
                return;
            }
            let clip_min = vec2<f32>(clip_min_x_at(layer), clip_min_y_at(layer));
            let clip_max = vec2<f32>(clip_max_x_at(layer), clip_max_y_at(layer));
            min_xy = max(min_xy, clip_min);
            max_xy = min(max_xy, clip_max);
            if (max_xy.x <= min_xy.x || max_xy.y <= min_xy.y) {
                var clipped_v: VertexOut;
                clipped_v.clip_pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
                clipped_v.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                clipped_v.uv = local_uv;
                clipped_v.world_pos = min_xy;
                clipped_v.cmd_type = cmd.cmd_type;
                clipped_v.flags = cmd.flags;
                clipped_v.state_index = cmd.state_index;
                clipped_v._pad0 = 0u;
                clipped_v.extra0 = extra0;
                clipped_v.extra1 = extra1;
                clipped_v.extra2 = extra2;
                clipped_v.extra3 = extra3;
                vertices[vertex_index] = clipped_v;
                return;
            }
            if (parent == FS_CLIP_LAYER_NONE) {
                break;
            }
            layer = parent;
            parent = clip_parent_at(parent);
            depth = depth + 1u;
            if (depth >= FS_CLIP_LAYER_COUNT) {
                break;
            }
        }
    }

    if (use_oriented_quad) {
        let det = quad_du.x * quad_dv.y - quad_du.y * quad_dv.x;
        if (abs(det) <= 1e-8) {
            var clipped_v: VertexOut;
            clipped_v.clip_pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
            clipped_v.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            clipped_v.uv = local_uv;
            clipped_v.world_pos = min_xy;
            clipped_v.cmd_type = cmd.cmd_type;
            clipped_v.flags = cmd.flags;
            clipped_v.state_index = cmd.state_index;
            clipped_v._pad0 = 0u;
            clipped_v.extra0 = extra0;
            clipped_v.extra1 = extra1;
            clipped_v.extra2 = extra2;
            clipped_v.extra3 = extra3;
            vertices[vertex_index] = clipped_v;
            return;
        }
        let inv_det = 1.0 / det;
        let c0 = oriented_uv_from_world(vec2<f32>(min_xy.x, min_xy.y), quad_origin, quad_du, quad_dv, inv_det);
        let c1 = oriented_uv_from_world(vec2<f32>(max_xy.x, min_xy.y), quad_origin, quad_du, quad_dv, inv_det);
        let c2 = oriented_uv_from_world(vec2<f32>(min_xy.x, max_xy.y), quad_origin, quad_du, quad_dv, inv_det);
        let c3 = oriented_uv_from_world(vec2<f32>(max_xy.x, max_xy.y), quad_origin, quad_du, quad_dv, inv_det);
        let uv_min_raw = min(min(c0, c1), min(c2, c3));
        let uv_max_raw = max(max(c0, c1), max(c2, c3));
        quad_uv_min = clamp(uv_min_raw, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
        quad_uv_max = clamp(uv_max_raw, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
        if (quad_uv_max.x <= quad_uv_min.x || quad_uv_max.y <= quad_uv_min.y) {
            var clipped_v: VertexOut;
            clipped_v.clip_pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
            clipped_v.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            clipped_v.uv = local_uv;
            clipped_v.world_pos = min_xy;
            clipped_v.cmd_type = cmd.cmd_type;
            clipped_v.flags = cmd.flags;
            clipped_v.state_index = cmd.state_index;
            clipped_v._pad0 = 0u;
            clipped_v.extra0 = extra0;
            clipped_v.extra1 = extra1;
            clipped_v.extra2 = extra2;
            clipped_v.extra3 = extra3;
            vertices[vertex_index] = clipped_v;
            return;
        }
    }

    let safe_size = max(max_xy - min_xy, vec2<f32>(1.0, 1.0));
    var out_uv = local_uv;
    var world = min_xy + local_uv * safe_size;
    if (use_oriented_quad) {
        out_uv = quad_uv_min + local_uv * (quad_uv_max - quad_uv_min);
        world = quad_origin + quad_du * out_uv.x + quad_dv * out_uv.y;
    }

    var out_flags = cmd.flags;
    if ((state.clip_meta.x & CLIP_PATH_BIT) != 0u) {
        out_flags = out_flags | FS_RENDER_FLAG_CLIP_MASK;
        if (state.clip_meta.y < FS_CLIP_LAYER_COUNT) {
            out_flags = out_flags |
                ((state.clip_meta.y << FS_RENDER_FLAG_CLIP_LAYER_SHIFT) & FS_RENDER_FLAG_CLIP_LAYER_MASK);
        }
        if (state.clip_meta.z != FS_CLIP_LAYER_NONE && state.clip_meta.z < 127u) {
            out_flags = out_flags |
                (((state.clip_meta.z + 1u) << FS_RENDER_FLAG_CLIP_PARENT_SHIFT) & FS_RENDER_FLAG_CLIP_PARENT_MASK);
        }
    }
    var out_color = unpack4x8unorm(cmd.color_rgba8);
    out_color.a = out_color.a * clamp(state.xform1.w, 0.0, 1.0);

    var out_v: VertexOut;
    out_v.clip_pos = clip_from_world(world, uniforms.viewport);
    out_v.color = out_color;
    out_v.uv = out_uv;
    out_v.world_pos = world;
    out_v.cmd_type = cmd.cmd_type;
    out_v.flags = out_flags;
    out_v.state_index = cmd.state_index;
    out_v._pad0 = 0u;
    out_v.extra0 = extra0;
    out_v.extra1 = extra1;
    out_v.extra2 = extra2;
    out_v.extra3 = extra3;
    vertices[vertex_index] = out_v;
}
);

static const char* FS_RENDER_WGSL = WGSL_CODE(
const CMD_RECT: u32 = 0u;
const CMD_IMAGE: u32 = 1u;
const CMD_TEXT: u32 = 2u;
const CMD_LINE: u32 = 3u;
const CMD_PATH_SEGMENT: u32 = 4u;
const CMD_CIRCLE: u32 = 5u;
const CMD_ARC: u32 = 6u;
const CMD_BEZIER_QUAD: u32 = 7u;
const CMD_RECT_STROKE: u32 = 8u;
const CMD_ELLIPSE: u32 = 9u;
const CMD_BEZIER_CUBIC: u32 = 10u;
const CMD_TRIANGLE: u32 = 11u;
const FS_PREMULTIPLIED_OUTPUT: bool = true;
const FS_TEXT_FLAG_COLOR_GLYPH: u32 = 1u;
const FS_TEXT_FLAG_STROKE: u32 = 2u;
const FS_TRI_FLAG_AA_EDGE0: u32 = 1u << 2u;
const FS_TRI_FLAG_AA_EDGE1: u32 = 1u << 3u;
const FS_TRI_FLAG_AA_EDGE2: u32 = 1u << 4u;
const FS_TRI_FLAG_AA_MASK: u32 = FS_TRI_FLAG_AA_EDGE0 | FS_TRI_FLAG_AA_EDGE1 | FS_TRI_FLAG_AA_EDGE2;
const FS_LINE_FLAG_BUTT: u32 = 1u << 5u;
const FS_LINE_FLAG_NO_AA_START: u32 = 1u << 6u;
const FS_LINE_FLAG_NO_AA_END: u32 = 1u << 7u;
const FS_RENDER_FLAG_SHADOW_BLUR_SHIFT: u32 = 8u;
const FS_RENDER_FLAG_SHADOW_BLUR_MASK: u32 = 0xFu << FS_RENDER_FLAG_SHADOW_BLUR_SHIFT;
const FS_RENDER_FLAG_SHADOW: u32 = 1u << 12u;
const FS_RENDER_FLAG_PATTERN_SHADE: u32 = 1u << 13u;
const FS_RENDER_FLAG_IMAGE_NEAREST: u32 = 1u << 1u;
const FS_RENDER_FLAG_CONTINUOUS_CORNER: u32 = 1u << 1u;
const FS_RENDER_FLAG_ORIENTED_QUAD: u32 = 1u << 14u;
const FS_RENDER_FLAG_CLIP_MASK: u32 = 1u << 16u;
const FS_RENDER_FLAG_CLIP_PARENT_SHIFT: u32 = 17u;
const FS_RENDER_FLAG_CLIP_PARENT_MASK: u32 = 0x7Fu << FS_RENDER_FLAG_CLIP_PARENT_SHIFT;
const FS_RENDER_FLAG_CLIP_LAYER_SHIFT: u32 = 24u;
const FS_RENDER_FLAG_CLIP_LAYER_MASK: u32 = 0xFFu << FS_RENDER_FLAG_CLIP_LAYER_SHIFT;
const FS_CLIP_LAYER_COUNT: u32 = 64u;
const FS_CLIP_LAYER_NONE: u32 = 0xFFFFFFFFu;

struct Uniforms {
    viewport: vec2<f32>,
    command_count: u32,
    clip_enabled: u32,
    clip_min: vec2<f32>,
    clip_max: vec2<f32>,
};

struct ClipLayers {
    parent: array<vec4<u32>, 16>,
    min_x: array<vec4<u32>, 16>,
    min_y: array<vec4<u32>, 16>,
    max_x: array<vec4<u32>, 16>,
    max_y: array<vec4<u32>, 16>,
};

struct CommandState {
    clip_rect: vec4<f32>,
    clip_meta: vec4<u32>,
    xform0: vec4<f32>,
    xform1: vec4<f32>,
    pattern_inv0: vec4<f32>,
    pattern_inv1: vec4<f32>,
    pattern_meta: vec4<f32>,
};

@group(0) @binding(0) var img_tex: texture_2d_array<f32>;
@group(0) @binding(1) var img_samp: sampler;
@group(0) @binding(2) var glyph_tex: texture_2d<f32>;
@group(0) @binding(3) var glyph_samp: sampler;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;
@group(0) @binding(5) var clip_tex: texture_2d_array<f32>;
@group(0) @binding(6) var clip_samp: sampler;
@group(0) @binding(7) var<uniform> clip_layers: ClipLayers;
@group(0) @binding(8) var<storage, read> command_states: array<CommandState>;

fn lane_pick_u32(v: vec4<u32>, lane: u32) -> u32 {
    switch (lane) {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        case 2u: { return v.z; }
        default: { return v.w; }
    }
}

fn clip_parent_at(layer: u32) -> u32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return lane_pick_u32(clip_layers.parent[slot], lane);
}

fn clip_min_x_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.min_x[slot], lane));
}

fn clip_min_y_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.min_y[slot], lane));
}

fn clip_max_x_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.max_x[slot], lane));
}

fn clip_max_y_at(layer: u32) -> f32 {
    let slot = layer >> 2u;
    let lane = layer & 3u;
    return f32(lane_pick_u32(clip_layers.max_y[slot], lane));
}

struct VSIn {
    @location(0) clip_pos: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_pos: vec2<f32>,
    @location(4) @interpolate(flat) cmd_type: u32,
    @location(5) @interpolate(flat) flags: u32,
    @location(6) @interpolate(flat) state_index: u32,
    @location(7) extra0: vec4<f32>,
    @location(8) extra1: vec4<f32>,
    @location(9) extra2: vec4<f32>,
    @location(10) extra3: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec2<f32>,
    @location(3) @interpolate(flat) cmd_type: u32,
    @location(4) @interpolate(flat) flags: u32,
    @location(5) @interpolate(flat) state_index: u32,
    @location(6) extra0: vec4<f32>,
    @location(7) extra1: vec4<f32>,
    @location(8) extra2: vec4<f32>,
    @location(9) extra3: vec4<f32>,
};

fn sdf_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

fn rounded_rect_sdf(p: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let b = max(size * 0.5 - vec2<f32>(radius, radius), vec2<f32>(0.0, 0.0));
    let q = abs(p) - b;
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn lane_f32(v: vec4<f32>, lane: u32) -> f32 {
    switch (lane) {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        case 2u: { return v.z; }
        default: { return v.w; }
    }
}

fn round_rect_corner_index(p: vec2<f32>) -> u32 {
    if (p.y < 0.0) {
        return select(0u, 1u, p.x >= 0.0);
    }
    return select(3u, 2u, p.x >= 0.0);
}

fn superellipse_point(u: f32, radius: vec2<f32>, exponent: f32) -> vec2<f32> {
    let x = clamp(u, 0.0, 1.0);
    let y = pow(max(1.0 - pow(x, exponent), 0.0), 1.0 / exponent);
    return vec2<f32>(radius.x * x, radius.y * y);
}

fn superellipse_signed_distance(q: vec2<f32>, radius: vec2<f32>, exponent: f32) -> f32 {
    // Golden-section search gives the actual Euclidean distance to the quarter
    // superellipse. Unlike F/|grad F|, this remains continuous where the curved
    // corner meets either straight edge, including offset contours used by stroke.
    let ratio = 0.6180339887498948;
    var lo = 0.0;
    var hi = 1.0;
    var u0 = hi - (hi - lo) * ratio;
    var u1 = lo + (hi - lo) * ratio;
    var p0 = superellipse_point(u0, radius, exponent);
    var p1 = superellipse_point(u1, radius, exponent);
    var v0 = q - p0;
    var v1 = q - p1;
    var d0 = dot(v0, v0);
    var d1 = dot(v1, v1);

    for (var i: u32 = 0u; i < 18u; i = i + 1u) {
        if (d0 <= d1) {
            hi = u1;
            u1 = u0;
            d1 = d0;
            u0 = hi - (hi - lo) * ratio;
            p0 = superellipse_point(u0, radius, exponent);
            v0 = q - p0;
            d0 = dot(v0, v0);
        } else {
            lo = u0;
            u0 = u1;
            d0 = d1;
            u1 = lo + (hi - lo) * ratio;
            p1 = superellipse_point(u1, radius, exponent);
            v1 = q - p1;
            d1 = dot(v1, v1);
        }
    }

    let top_delta = q - vec2<f32>(0.0, radius.y);
    let right_delta = q - vec2<f32>(radius.x, 0.0);
    let distance_sq = min(min(d0, d1),
                          min(dot(top_delta, top_delta), dot(right_delta, right_delta)));
    let normalized = max(q / radius, vec2<f32>(0.0, 0.0));
    let implicit = pow(normalized.x, exponent) + pow(normalized.y, exponent) - 1.0;
    let distance = sqrt(max(distance_sq, 0.0));
    return select(-distance, distance, implicit >= 0.0);
}

fn round_rect_profile_sdf(
    p: vec2<f32>, size: vec2<f32>, radii_x: vec4<f32>, radii_y: vec4<f32>,
    continuous: bool
) -> f32 {
    let lane = round_rect_corner_index(p);
    let radius = max(vec2<f32>(lane_f32(radii_x, lane), lane_f32(radii_y, lane)),
                     vec2<f32>(0.0, 0.0));
    let half_size = max(size * 0.5, vec2<f32>(0.0, 0.0));
    let corner_center = max(half_size - radius, vec2<f32>(0.0, 0.0));
    let q = abs(p) - corner_center;
    if (radius.x <= 1e-5 || radius.y <= 1e-5) {
        return max(abs(p).x - half_size.x, abs(p).y - half_size.y);
    }
    if (q.x <= 0.0 || q.y <= 0.0) {
        return max(q.x - radius.x, q.y - radius.y);
    }
    var exponent = 2.0;
    if (continuous) {
        let extent = max(
            (radius.x * 2.0) / max(size.x, 1e-5),
            (radius.y * 2.0) / max(size.y, 1e-5)
        );
        exponent = mix(4.0, 2.0, smoothstep(0.82, 1.0, extent));
    }
    return superellipse_signed_distance(q, radius, exponent);
}

fn round_rect_local_point(input: VSOut) -> vec2<f32> {
    if ((input.flags & FS_RENDER_FLAG_ORIENTED_QUAD) == 0u) {
        return input.world_pos - input.extra0.xy;
    }
    let state = command_states[input.state_index];
    let a = state.xform0.x;
    let b = state.xform0.y;
    let c = state.xform0.z;
    let d = state.xform0.w;
    let translated = input.world_pos - state.xform1.xy;
    let det = a * d - b * c;
    if (abs(det) <= 1e-8) {
        return vec2<f32>(1e9, 1e9);
    }
    let local = vec2<f32>(d * translated.x - c * translated.y,
                          -b * translated.x + a * translated.y) / det;
    return local - input.extra0.xy;
}

fn in_arc(angle: f32, start_angle: f32, end_angle: f32) -> bool {
    let two_pi = 6.283185307;
    var a = angle;
    var s = start_angle;
    var e = end_angle;
    if (a < 0.0) { a = a + two_pi; }
    if (s < 0.0) { s = s + two_pi; }
    if (e < 0.0) { e = e + two_pi; }
    if (s <= e) {
        return a >= s && a <= e;
    }
    return a >= s || a <= e;
}

fn tri_edge_alpha(sd: f32, aa: f32, use_aa: bool) -> f32 {
    if (use_aa) {
        return smoothstep(-aa, aa, sd);
    }
    let hard_bias = aa * 0.35;
    return select(0.0, 1.0, sd >= -hard_bias);
}

fn rounded_rect_fill_alpha(uv: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let p = (uv - vec2<f32>(0.5, 0.5)) * size;
    let d = rounded_rect_sdf(p, size, radius);
    let aa = max(fwidth(d), 1.0);
    return 1.0 - smoothstep(-aa, aa, d);
}

fn rounded_rect_stroke_alpha(uv: vec2<f32>, size: vec2<f32>, radius: f32, stroke_w: f32) -> f32 {
    let p = (uv - vec2<f32>(0.5, 0.5)) * size;
    let d = rounded_rect_sdf(p, size, radius);
    let ring = stroke_w * 0.5 - abs(d);
    let aa = max(fwidth(ring), 1.0);
    return smoothstep(-aa, aa, ring);
}

fn rounded_rect_fill_alpha_4tap(uv: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let du = dpdx(uv);
    let dv = dpdy(uv);
    let o = 0.375;
    let a0 = rounded_rect_fill_alpha(uv + du * -o + dv * -o, size, radius);
    let a1 = rounded_rect_fill_alpha(uv + du *  o + dv * -o, size, radius);
    let a2 = rounded_rect_fill_alpha(uv + du * -o + dv *  o, size, radius);
    let a3 = rounded_rect_fill_alpha(uv + du *  o + dv *  o, size, radius);
    return (a0 + a1 + a2 + a3) * 0.25;
}

fn rounded_rect_stroke_alpha_4tap(uv: vec2<f32>, size: vec2<f32>, radius: f32, stroke_w: f32) -> f32 {
    let du = dpdx(uv);
    let dv = dpdy(uv);
    let o = 0.375;
    let a0 = rounded_rect_stroke_alpha(uv + du * -o + dv * -o, size, radius, stroke_w);
    let a1 = rounded_rect_stroke_alpha(uv + du *  o + dv * -o, size, radius, stroke_w);
    let a2 = rounded_rect_stroke_alpha(uv + du * -o + dv *  o, size, radius, stroke_w);
    let a3 = rounded_rect_stroke_alpha(uv + du *  o + dv *  o, size, radius, stroke_w);
    return (a0 + a1 + a2 + a3) * 0.25;
}

fn shadow_alpha_curve(alpha: f32, blur: f32) -> f32 {
    let b = clamp(blur / 15.0, 0.0, 1.0);
    let expanded = clamp(alpha + 0.08 * b, 0.0, 1.0);
    let gamma = mix(1.0, 0.78, b);
    return clamp(pow(expanded, gamma), 0.0, 1.0);
}

fn repeat_coord(v: f32, span: f32) -> f32 {
    return v - floor(v / span) * span;
}

fn sample_image_rgba(uv: vec2<f32>, layer: i32, flags: u32, quality_hint: f32) -> vec4<f32> {
    let dims_u = textureDimensions(img_tex, 0);
    let dims_f = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
    if ((flags & FS_RENDER_FLAG_IMAGE_NEAREST) != 0u) {
        let max_coord = vec2<i32>(i32(dims_u.x) - 1, i32(dims_u.y) - 1);
        let texel = clamp(vec2<i32>(floor(uv * dims_f)), vec2<i32>(0, 0), max_coord);
        return textureLoad(img_tex, texel, layer, 0);
    }
    let q = u32(clamp(round(quality_hint), 0.0, 2.0));
    let base = textureSample(img_tex, img_samp, uv, layer);
    if (q != 2u) {
        return base;
    }
    // Apply HQ filter only on minification to avoid over-softening magnified content.
    let grad_x = dpdx(uv) * dims_f;
    let grad_y = dpdy(uv) * dims_f;
    let footprint = max(length(grad_x), length(grad_y));
    if (footprint <= 1.0) {
        return base;
    }
    let texel = 1.0 / max(dims_f, vec2<f32>(1.0, 1.0));
    let ox = vec2<f32>(texel.x * 0.35, 0.0);
    let oy = vec2<f32>(0.0, texel.y * 0.35);
    let s1 = textureSample(img_tex, img_samp, uv + ox, layer);
    let s2 = textureSample(img_tex, img_samp, uv - ox, layer);
    let s3 = textureSample(img_tex, img_samp, uv + oy, layer);
    let s4 = textureSample(img_tex, img_samp, uv - oy, layer);
    return base * 0.5 + (s1 + s2 + s3 + s4) * 0.125;
}

fn sample_pattern_rgba(state: CommandState, world_pos: vec2<f32>, flags: u32) -> vec4<f32> {
    let pw = state.pattern_meta.x;
    let ph = state.pattern_meta.y;
    if (pw <= 0.0 || ph <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let repeat_mode = u32(clamp(round(state.pattern_meta.w), 0.0, 3.0));
    let repeat_x = repeat_mode == 0u || repeat_mode == 1u;
    let repeat_y = repeat_mode == 0u || repeat_mode == 2u;
    let px = state.pattern_inv0.x * world_pos.x + state.pattern_inv0.z * world_pos.y + state.pattern_inv1.x;
    let py = state.pattern_inv0.y * world_pos.x + state.pattern_inv0.w * world_pos.y + state.pattern_inv1.y;
    var sx = floor(px);
    var sy = floor(py);
    if (!repeat_x && (sx < 0.0 || sx >= pw)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (!repeat_y && (sy < 0.0 || sy >= ph)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (repeat_x) {
        sx = repeat_coord(sx, pw);
    }
    if (repeat_y) {
        sy = repeat_coord(sy, ph);
    }
    let atlas_xy = vec2<f32>(state.pattern_inv1.z + sx + 0.5, state.pattern_inv1.w + sy + 0.5);
    let tex_dims_u = textureDimensions(img_tex, 0);
    let tex_dims = vec2<f32>(f32(tex_dims_u.x), f32(tex_dims_u.y));
    let uv = atlas_xy / max(tex_dims, vec2<f32>(1.0, 1.0));
    let layer = i32(clamp(round(state.pattern_meta.z), 0.0, 3.0));
    return sample_image_rgba(uv, layer, flags, state.xform1.z);
}

@vertex
fn vs_main(input: VSIn) -> VSOut {
    var out_v: VSOut;
    out_v.position = input.clip_pos;
    out_v.color = input.color;
    out_v.uv = input.uv;
    out_v.world_pos = input.world_pos;
    out_v.cmd_type = input.cmd_type;
    out_v.flags = input.flags;
    out_v.state_index = input.state_index;
    out_v.extra0 = input.extra0;
    out_v.extra1 = input.extra1;
    out_v.extra2 = input.extra2;
    out_v.extra3 = input.extra3;
    return out_v;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    var clip_alpha_mask = 1.0;
    if ((input.flags & FS_RENDER_FLAG_CLIP_MASK) != 0u) {
        let uv = input.world_pos / uniforms.viewport;
        if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
            discard;
        }
        let layer_bits = (input.flags & FS_RENDER_FLAG_CLIP_LAYER_MASK) >> FS_RENDER_FLAG_CLIP_LAYER_SHIFT;
        let parent_bits = (input.flags & FS_RENDER_FLAG_CLIP_PARENT_MASK) >> FS_RENDER_FLAG_CLIP_PARENT_SHIFT;
        var layer = layer_bits;
        var parent = select(FS_CLIP_LAYER_NONE, parent_bits - 1u, parent_bits != 0u);
        var depth: u32 = 0u;
        loop {
            if (layer >= FS_CLIP_LAYER_COUNT) {
                discard;
            }
            let min_x = clip_min_x_at(layer);
            let min_y = clip_min_y_at(layer);
            let max_x = clip_max_x_at(layer);
            let max_y = clip_max_y_at(layer);
            if (input.world_pos.x < min_x || input.world_pos.y < min_y ||
                input.world_pos.x >= max_x || input.world_pos.y >= max_y) {
                discard;
            }
            let clip_alpha = textureSample(clip_tex, clip_samp, uv, i32(layer)).r;
            clip_alpha_mask = clip_alpha_mask * clip_alpha;
            if (clip_alpha_mask < 0.001) {
                discard;
            }
            if (parent == FS_CLIP_LAYER_NONE) {
                break;
            }
            layer = parent;
            parent = clip_parent_at(parent);
            depth = depth + 1u;
            if (depth >= FS_CLIP_LAYER_COUNT) {
                break;
            }
        }
    }
    var out_color = input.color;
    let cmd_state = command_states[input.state_index];
    var alpha = 1.0;
    let shadow_blur_q = (input.flags & FS_RENDER_FLAG_SHADOW_BLUR_MASK) >> FS_RENDER_FLAG_SHADOW_BLUR_SHIFT;
    let shadow_blur = f32(shadow_blur_q);
    let is_shadow = (input.flags & FS_RENDER_FLAG_SHADOW) != 0u;

    switch (input.cmd_type) {
        case CMD_RECT: {
            let size = input.extra0.zw;
            let p = round_rect_local_point(input);
            let continuous = (input.flags & FS_RENDER_FLAG_CONTINUOUS_CORNER) != 0u;
            let d = round_rect_profile_sdf(p, size, input.extra1, input.extra2, continuous);
            if (is_shadow) {
                let aa = max(fwidth(d), 1.0) + shadow_blur;
                alpha = 1.0 - smoothstep(-aa, aa, d);
            } else {
                let aa = max(fwidth(d), 0.65);
                alpha = 1.0 - smoothstep(-aa, aa, d);
            }
            break;
        }
        case CMD_RECT_STROKE: {
            let size = input.extra0.zw;
            let stroke_w = max(input.extra3.x, 0.0);
            let p = round_rect_local_point(input);
            let continuous = (input.flags & FS_RENDER_FLAG_CONTINUOUS_CORNER) != 0u;
            let d = round_rect_profile_sdf(p, size, input.extra1, input.extra2, continuous);
            let ring = stroke_w * 0.5 - abs(d);
            if (is_shadow) {
                let aa = max(fwidth(ring), 1.0) + shadow_blur;
                alpha = smoothstep(-aa, aa, ring);
            } else {
                let aa = max(fwidth(ring), 0.65);
                alpha = smoothstep(-aa, aa, ring);
            }
            break;
        }
        case CMD_IMAGE: {
            let uv = input.extra1.xy + input.uv * input.extra1.zw;
            let layer = i32(clamp(input.extra2.x, 0.0, 3.0));
            let sampled = sample_image_rgba(uv, layer, input.flags, cmd_state.xform1.z);
            if (is_shadow) {
                var image_alpha = sampled.a;
                if (shadow_blur > 0.5) {
                    let draw_size = max(input.extra0.zw, vec2<f32>(1.0, 1.0));
                    let uv_step = (input.extra1.zw / draw_size) * max(shadow_blur * 0.5, 0.5);
                    let ax0 = sample_image_rgba(uv + vec2<f32>( uv_step.x, 0.0), layer, input.flags, cmd_state.xform1.z).a;
                    let ax1 = sample_image_rgba(uv + vec2<f32>(-uv_step.x, 0.0), layer, input.flags, cmd_state.xform1.z).a;
                    let ay0 = sample_image_rgba(uv + vec2<f32>(0.0,  uv_step.y), layer, input.flags, cmd_state.xform1.z).a;
                    let ay1 = sample_image_rgba(uv + vec2<f32>(0.0, -uv_step.y), layer, input.flags, cmd_state.xform1.z).a;
                    let d0 = sample_image_rgba(uv + vec2<f32>( uv_step.x,  uv_step.y), layer, input.flags, cmd_state.xform1.z).a;
                    let d1 = sample_image_rgba(uv + vec2<f32>(-uv_step.x,  uv_step.y), layer, input.flags, cmd_state.xform1.z).a;
                    let d2 = sample_image_rgba(uv + vec2<f32>( uv_step.x, -uv_step.y), layer, input.flags, cmd_state.xform1.z).a;
                    let d3 = sample_image_rgba(uv + vec2<f32>(-uv_step.x, -uv_step.y), layer, input.flags, cmd_state.xform1.z).a;
                    image_alpha =
                        image_alpha * 0.25 +
                        (ax0 + ax1 + ay0 + ay1) * 0.125 +
                        (d0 + d1 + d2 + d3) * 0.0625;
                }
                alpha = image_alpha;
                break;
            }
            var image_color = sampled * out_color;
            image_color.a = image_color.a * clip_alpha_mask;
            if (image_color.a < 0.01) {
                discard;
            }
            return image_color;
        }
        case CMD_TEXT: {
            let uv = input.extra1.xy + input.uv * input.extra1.zw;
            let sampled = textureSampleBias(glyph_tex, glyph_samp, uv, -0.20);
            if ((input.flags & FS_TEXT_FLAG_COLOR_GLYPH) != 0u) {
                if ((input.flags & FS_TEXT_FLAG_STROKE) != 0u) {
                    discard;
                }
                if (is_shadow) {
                    var glyph_alpha = sampled.a;
                    if (shadow_blur > 0.5) {
                        let draw_size = max(input.extra0.zw, vec2<f32>(1.0, 1.0));
                        let uv_step = (input.extra1.zw / draw_size) * max(shadow_blur * 0.5, 0.5);
                        let ax0 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>( uv_step.x, 0.0), -0.20).a;
                        let ax1 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>(-uv_step.x, 0.0), -0.20).a;
                        let ay0 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>(0.0,  uv_step.y), -0.20).a;
                        let ay1 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>(0.0, -uv_step.y), -0.20).a;
                        let d0 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>( uv_step.x,  uv_step.y), -0.20).a;
                        let d1 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>(-uv_step.x,  uv_step.y), -0.20).a;
                        let d2 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>( uv_step.x, -uv_step.y), -0.20).a;
                        let d3 = textureSampleBias(glyph_tex, glyph_samp, uv + vec2<f32>(-uv_step.x, -uv_step.y), -0.20).a;
                        glyph_alpha =
                            glyph_alpha * 0.25 +
                            (ax0 + ax1 + ay0 + ay1) * 0.125 +
                            (d0 + d1 + d2 + d3) * 0.0625;
                    }
                    alpha = glyph_alpha;
                    break;
                }
                var color_glyph = vec4<f32>(sampled.rgb, sampled.a * out_color.a * clip_alpha_mask);
                if (color_glyph.a < 0.01) {
                    discard;
                }
                if (FS_PREMULTIPLIED_OUTPUT) {
                    color_glyph = vec4<f32>(color_glyph.rgb * color_glyph.a, color_glyph.a);
                }
                return color_glyph;
            }
            // Match use.gpu-style sharper readback by slightly biasing towards higher mip detail.
            let sdf = sampled.a;
            let sdf_onedge = input.extra2.y;
            let px_dist_scale = max(input.extra2.z, 1e-3);
            let tex_size_u = textureDimensions(glyph_tex, 0);
            let tex_size = vec2<f32>(f32(tex_size_u.x), f32(tex_size_u.y));
            let atlas_w = max(input.extra1.z * tex_size.x, 1.0);
            let atlas_h = max(input.extra1.w * tex_size.y, 1.0);
            let texel_span = max(
                max(fwidth(input.uv.x) * atlas_w, fwidth(input.uv.y) * atlas_h),
                1e-3
            );
            // Both stb_truetype and the FreeType backend encode SDF texels as
            //   encoded_u8 = onedge_u8 + signed_distance_px * pixel_dist_scale.
            // Decode the normalized texture value back to bake-space pixels,
            // then convert bake-space pixels to screen-space pixels.
            let sd_bake_px = (sdf - sdf_onedge) * (255.0 / px_dist_scale);
            let sd = sd_bake_px / texel_span;
            // sd is already measured in screen pixels. A one-pixel transition is
            // therefore explicit and stable; fwidth(sd) is invalid at the glyph
            // quad boundary because helper fragments sample outside the atlas slot.
            let w = 0.5;
            if ((input.flags & FS_TEXT_FLAG_STROKE) != 0u) {
                let stroke_width = max(input.extra2.w, 0.0);
                if (stroke_width <= 1e-4) {
                    discard;
                }
                let half_sw = stroke_width * 0.5;
                let dist_edge = abs(sd);
                let sw = w + select(0.0, shadow_blur, is_shadow);
                alpha = 1.0 - smoothstep(half_sw - sw, half_sw + sw, dist_edge);
            } else {
                let sw = w + select(0.0, shadow_blur * 0.5, is_shadow);
                alpha = smoothstep(-sw, sw, sd);
            }
            break;
        }
        case CMD_LINE, CMD_PATH_SEGMENT: {
            let a = input.extra0.xy;
            let b = input.extra0.zw;
            let width = max(input.extra1.x, 1.0);
            let use_butt = (input.flags & FS_LINE_FLAG_BUTT) != 0u;
            if (use_butt) {
                let ba = b - a;
                let len = max(length(ba), 1e-6);
                let dir = ba / len;
                let n = vec2<f32>(-dir.y, dir.x);
                let mid = (a + b) * 0.5;
                let rel = input.world_pos - mid;
                let local = vec2<f32>(dot(rel, dir), dot(rel, n));
                let half_ext = vec2<f32>(len * 0.5, width * 0.5);
                let aa = max(length(vec2<f32>(fwidth(input.world_pos.x), fwidth(input.world_pos.y))), 0.75) + shadow_blur;
                let use_aa_start = (input.flags & FS_LINE_FLAG_NO_AA_START) == 0u;
                let use_aa_end = (input.flags & FS_LINE_FLAG_NO_AA_END) == 0u;
                let dist_side = half_ext.y - abs(local.y);
                let dist_start = local.x + half_ext.x;
                let dist_end = half_ext.x - local.x;
                let side_alpha = smoothstep(-aa, aa, dist_side);
                let start_alpha = tri_edge_alpha(dist_start, aa, use_aa_start);
                let end_alpha = tri_edge_alpha(dist_end, aa, use_aa_end);
                alpha = min(side_alpha, min(start_alpha, end_alpha));
            } else {
                let dist = sdf_segment(input.world_pos, a, b);
                let sdf = width * 0.5 - dist;
                let aa = 1.0 + shadow_blur;
                alpha = smoothstep(-aa, aa, sdf);
            }
            break;
        }
        case CMD_CIRCLE: {
            let c = input.extra0.xy;
            let r = input.extra0.z;
            let d = r - length(input.world_pos - c);
            let aa = 1.0 + shadow_blur;
            alpha = smoothstep(-aa, aa, d);
            break;
        }
        case CMD_ELLIPSE: {
            let c = input.extra0.xy;
            let rx = max(input.extra0.z, 1.0);
            let ry = max(input.extra0.w, 1.0);
            let p = (input.world_pos - c) / vec2<f32>(rx, ry);
            let d = 1.0 - length(p);
            let aa = (1.0 + shadow_blur) / min(rx, ry);
            alpha = smoothstep(-aa, aa, d);
            break;
        }
        case CMD_ARC: {
            let c = input.extra0.xy;
            let radius = input.extra0.z;
            let thickness = max(input.extra0.w, 1.0);
            let start_angle = input.extra1.x;
            let end_angle = input.extra1.y;
            let p = input.world_pos - c;
            let dist = length(p);
            let ang = atan2(p.y, p.x);
            let ring = thickness * 0.5 - abs(dist - radius);
            if (in_arc(ang, start_angle, end_angle)) {
                let aa = 1.0 + shadow_blur;
                alpha = smoothstep(-aa, aa, ring);
            } else {
                alpha = 0.0;
            }
            break;
        }
        case CMD_BEZIER_QUAD: {
            let p0 = input.extra0.xy;
            let cp = input.extra0.zw;
            let p1 = input.extra1.xy;
            let width = max(input.extra1.z, 1.0);
            let est_len = length(cp - p0) + length(p1 - cp);
            let steps = i32(clamp(est_len / 10.0 + 6.0, 10.0, 64.0));
            var min_dist = 1e9;
            var prev = p0;
            for (var i: i32 = 1; i <= 64; i = i + 1) {
                if (i > steps) {
                    break;
                }
                let t = f32(i) / f32(steps);
                let mt = 1.0 - t;
                let curve = mt * mt * p0 + 2.0 * mt * t * cp + t * t * p1;
                let d = sdf_segment(input.world_pos, prev, curve);
                min_dist = min(min_dist, d);
                prev = curve;
            }
            let sdf = width * 0.5 - min_dist;
            let aa = max(length(vec2<f32>(fwidth(input.world_pos.x), fwidth(input.world_pos.y))), 0.75) + shadow_blur;
            alpha = smoothstep(-aa, aa, sdf);
            break;
        }
        case CMD_BEZIER_CUBIC: {
            let p0 = input.extra0.xy;
            let c0 = input.extra0.zw;
            let c1 = input.extra1.xy;
            let p1 = input.extra1.zw;
            let width = max(input.extra2.x, 1.0);
            let est_len = length(c0 - p0) + length(c1 - c0) + length(p1 - c1);
            let steps = i32(clamp(est_len / 8.0 + 8.0, 14.0, 96.0));
            var min_dist = 1e9;
            var prev = p0;
            for (var i: i32 = 1; i <= 96; i = i + 1) {
                if (i > steps) {
                    break;
                }
                let t = f32(i) / f32(steps);
                let mt = 1.0 - t;
                let curve = mt * mt * mt * p0 +
                            3.0 * mt * mt * t * c0 +
                            3.0 * mt * t * t * c1 +
                            t * t * t * p1;
                let d = sdf_segment(input.world_pos, prev, curve);
                min_dist = min(min_dist, d);
                prev = curve;
            }
            let sdf = width * 0.5 - min_dist;
            let aa = max(length(vec2<f32>(fwidth(input.world_pos.x), fwidth(input.world_pos.y))), 0.75) + shadow_blur;
            alpha = smoothstep(-aa, aa, sdf);
            break;
        }
        case CMD_TRIANGLE: {
            let a = input.extra0.xy;
            let b = input.extra0.zw;
            let c = input.extra1.xy;
            let p = input.world_pos;
            let e0 = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
            let e1 = (c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x);
            let e2 = (a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x);
            let area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            var orient = 1.0;
            if (area2 < 0.0) {
                orient = -1.0;
            }
            let d0 = orient * e0 / max(length(b - a), 1e-6);
            let d1 = orient * e1 / max(length(c - b), 1e-6);
            let d2 = orient * e2 / max(length(a - c), 1e-6);
            let aa = max(length(vec2<f32>(fwidth(p.x), fwidth(p.y))), 0.75) + shadow_blur;
            let tri_aa_flags = input.flags & FS_TRI_FLAG_AA_MASK;
            let a0 = tri_edge_alpha(d0, aa, (tri_aa_flags & FS_TRI_FLAG_AA_EDGE0) != 0u);
            let a1 = tri_edge_alpha(d1, aa, (tri_aa_flags & FS_TRI_FLAG_AA_EDGE1) != 0u);
            let a2 = tri_edge_alpha(d2, aa, (tri_aa_flags & FS_TRI_FLAG_AA_EDGE2) != 0u);
            alpha = min(a0, min(a1, a2));
            break;
        }
        default: {
            return vec4<f32>(1.0, 0.0, 1.0, 1.0);
        }
    }

    if ((input.flags & FS_RENDER_FLAG_PATTERN_SHADE) != 0u) {
        let pattern_rgba = sample_pattern_rgba(cmd_state, input.world_pos, input.flags);
        if (pattern_rgba.a <= 0.001) {
            discard;
        }
        out_color = vec4<f32>(pattern_rgba.rgb, out_color.a * pattern_rgba.a);
    }

    if (is_shadow) {
        alpha = shadow_alpha_curve(alpha, shadow_blur);
    }
    out_color.a = out_color.a * alpha;
    out_color.a = out_color.a * clip_alpha_mask;
    if (out_color.a < 0.01) {
        discard;
    }
    if (FS_PREMULTIPLIED_OUTPUT) {
        out_color = vec4<f32>(out_color.rgb * out_color.a, out_color.a);
    }
    return out_color;
}
);

static const char* FS_CLIP_EDGE_TRANSFORM_WGSL = WGSL_CODE(
struct ClipEdge {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
};

struct ClipJob {
    edge_offset: u32,
    edge_count: u32,
    layer: u32,
    fill_rule: u32,
    fill_min_x: u32,
    fill_min_y: u32,
    fill_max_x: u32,
    fill_max_y: u32,
    clear_min_x: u32,
    clear_min_y: u32,
    clear_max_x: u32,
    clear_max_y: u32,
    parent_layer: u32,
    has_parent: u32,
    scale_hint_bits: u32,
    fill_mode: u32,
};

struct ClipJobTransform {
    xform0: vec4<f32>,
    xform1: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> src_edges: array<ClipEdge>;
@group(0) @binding(1) var<storage, read> clip_jobs: array<ClipJob>;
@group(0) @binding(2) var<storage, read> clip_job_xforms: array<ClipJobTransform>;
@group(0) @binding(3) var<storage, read_write> dst_edges: array<ClipEdge>;

fn clip_transform_point(xf: ClipJobTransform, p: vec2<f32>) -> vec2<f32> {
    let a = xf.xform0.x;
    let b = xf.xform0.y;
    let c = xf.xform0.z;
    let d = xf.xform0.w;
    let e = xf.xform1.x;
    let f = xf.xform1.y;
    return vec2<f32>(a * p.x + c * p.y + e, b * p.x + d * p.y + f);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let job_index = gid.z;
    let job = clip_jobs[job_index];
    if (gid.x >= job.edge_count) {
        return;
    }
    let edge_index = job.edge_offset + gid.x;
    let in_edge = src_edges[edge_index];
    let xf = clip_job_xforms[job_index];
    let p0 = clip_transform_point(xf, vec2<f32>(in_edge.x0, in_edge.y0));
    let p1 = clip_transform_point(xf, vec2<f32>(in_edge.x1, in_edge.y1));
    dst_edges[edge_index] = ClipEdge(p0.x, p0.y, p1.x, p1.y);
}
);

static const char* FS_CLIP_MASK_WGSL = WGSL_CODE(
const FILL_RULE_NONZERO: u32 = 0u;
const FILL_RULE_EVENODD: u32 = 1u;
const CLIP_FILL_MODE_COVERAGE: u32 = 0u;
const CLIP_FILL_MODE_SDF: u32 = 1u;
const CLIP_FILL_MODE_ROUND_RECT: u32 = 2u;

struct ClipEdge {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
};

struct ClipJob {
    edge_offset: u32,
    edge_count: u32,
    layer: u32,
    fill_rule: u32,
    fill_min_x: u32,
    fill_min_y: u32,
    fill_max_x: u32,
    fill_max_y: u32,
    clear_min_x: u32,
    clear_min_y: u32,
    clear_max_x: u32,
    clear_max_y: u32,
    parent_layer: u32,
    has_parent: u32,
    scale_hint_bits: u32,
    fill_mode: u32,
};

struct ClipDispatch {
    job_count: u32,
    job_offset: u32,
    viewport_width: u32,
    viewport_height: u32,
    aa_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> clip_edges: array<ClipEdge>;
@group(0) @binding(1) var<storage, read> clip_jobs: array<ClipJob>;
@group(0) @binding(2) var clip_mask_out: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> clip_dispatch: ClipDispatch;

fn clip_point_in_evenodd(job: ClipJob, point: vec2<f32>) -> bool {
    var inside = false;
    for (var i: u32 = 0u; i < job.edge_count; i = i + 1u) {
        let e = clip_edges[job.edge_offset + i];
        let intersects =
            ((e.y0 > point.y) != (e.y1 > point.y)) &&
            (point.x < (e.x1 - e.x0) * (point.y - e.y0) / ((e.y1 - e.y0) + 1e-12) + e.x0);
        if (intersects) {
            inside = !inside;
        }
    }
    return inside;
}

fn clip_point_in_nonzero(job: ClipJob, point: vec2<f32>) -> bool {
    var winding: i32 = 0;
    for (var i: u32 = 0u; i < job.edge_count; i = i + 1u) {
        let e = clip_edges[job.edge_offset + i];
        if (e.y0 <= point.y) {
            if (e.y1 > point.y) {
                let left = (e.x1 - e.x0) * (point.y - e.y0) - (point.x - e.x0) * (e.y1 - e.y0);
                if (left > 0.0) {
                    winding = winding + 1;
                }
            }
        } else if (e.y1 <= point.y) {
            let left = (e.x1 - e.x0) * (point.y - e.y0) - (point.x - e.x0) * (e.y1 - e.y0);
            if (left < 0.0) {
                winding = winding - 1;
            }
        }
    }
    return winding != 0;
}

fn clip_point_in_path(job: ClipJob, point: vec2<f32>) -> bool {
    if (job.fill_rule == FILL_RULE_EVENODD) {
        return clip_point_in_evenodd(job, point);
    }
    return clip_point_in_nonzero(job, point);
}

fn clip_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

fn clip_sample_2x2(job: ClipJob, fx: f32, fy: f32) -> f32 {
    let s0 = clip_point_in_path(job, vec2<f32>(fx + 0.25, fy + 0.25));
    let s1 = clip_point_in_path(job, vec2<f32>(fx + 0.75, fy + 0.25));
    let s2 = clip_point_in_path(job, vec2<f32>(fx + 0.25, fy + 0.75));
    let s3 = clip_point_in_path(job, vec2<f32>(fx + 0.75, fy + 0.75));
    return 0.25 * (
        select(0.0, 1.0, s0) +
        select(0.0, 1.0, s1) +
        select(0.0, 1.0, s2) +
        select(0.0, 1.0, s3)
    );
}

fn clip_sample_4x4(job: ClipJob, fx: f32, fy: f32) -> f32 {
    var hits = 0.0;
    for (var sy: u32 = 0u; sy < 4u; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < 4u; sx = sx + 1u) {
            let sp = vec2<f32>(
                fx + (f32(sx) + 0.5) * 0.25,
                fy + (f32(sy) + 0.5) * 0.25
            );
            hits = hits + select(0.0, 1.0, clip_point_in_path(job, sp));
        }
    }
    return hits * (1.0 / 16.0);
}

fn clip_sample_8x8(job: ClipJob, fx: f32, fy: f32) -> f32 {
    var hits = 0.0;
    for (var sy: u32 = 0u; sy < 8u; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < 8u; sx = sx + 1u) {
            let sp = vec2<f32>(
                fx + (f32(sx) + 0.5) * 0.125,
                fy + (f32(sy) + 0.5) * 0.125
            );
            hits = hits + select(0.0, 1.0, clip_point_in_path(job, sp));
        }
    }
    return hits * (1.0 / 64.0);
}

fn clip_sample_sdf(job: ClipJob, fx: f32, fy: f32) -> f32 {
    let p = vec2<f32>(fx + 0.5, fy + 0.5);
    let inside = clip_point_in_path(job, p);
    var min_dist = 1e9;
    for (var i: u32 = 0u; i < job.edge_count; i = i + 1u) {
        let e = clip_edges[job.edge_offset + i];
        let d = clip_segment_distance(p, vec2<f32>(e.x0, e.y0), vec2<f32>(e.x1, e.y1));
        min_dist = min(min_dist, d);
    }
    let signed_dist = select(-min_dist, min_dist, inside);
    let aa = 0.85;
    return smoothstep(-aa, aa, signed_dist);
}

fn clip_sample_round_rect(job: ClipJob, fx: f32, fy: f32) -> f32 {
    let minp = vec2<f32>(f32(job.fill_min_x), f32(job.fill_min_y));
    let maxp = vec2<f32>(f32(job.fill_max_x), f32(job.fill_max_y));
    let size = max(maxp - minp, vec2<f32>(1.0, 1.0));
    let center = 0.5 * (minp + maxp);
    let p = vec2<f32>(fx + 0.5, fy + 0.5) - center;
    var r = max(bitcast<f32>(job.scale_hint_bits), 0.0);
    let r_max = 0.5 * min(size.x, size.y);
    if (r > r_max) {
        r = r_max;
    }
    let b = max(size * 0.5 - vec2<f32>(r, r), vec2<f32>(0.0, 0.0));
    let q = abs(p) - b;
    let d = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
    let aa = 0.85;
    return 1.0 - smoothstep(-aa, aa, d);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Job range is sliced by bind-group buffer range and z-dimension dispatch.
    let job_index = gid.z;
    let job = clip_jobs[job_index];

    let clear_w = select(0u, job.clear_max_x - job.clear_min_x, job.clear_max_x > job.clear_min_x);
    let clear_h = select(0u, job.clear_max_y - job.clear_min_y, job.clear_max_y > job.clear_min_y);
    if (gid.x >= clear_w || gid.y >= clear_h) {
        return;
    }

    let px = job.clear_min_x + gid.x;
    let py = job.clear_min_y + gid.y;

    let in_fill_rect =
        (px >= job.fill_min_x) && (py >= job.fill_min_y) &&
        (px < job.fill_max_x) && (py < job.fill_max_y);

    var a = 0.0;
    if (in_fill_rect && job.edge_count > 0u) {
        let fx = f32(px);
        let fy = f32(py);
        if (job.fill_mode == CLIP_FILL_MODE_SDF) {
            a = clip_sample_sdf(job, fx, fy);
        } else {
            // Coarse 2x2 classify first; only boundary pixels pay for dense sampling.
            let coarse = clip_sample_2x2(job, fx, fy);

            if (coarse <= 0.0) {
                a = 0.0;
            } else if (coarse >= 1.0) {
                a = 1.0;
            } else {
                // Boundary pixel: scale-aware supersampling level.
                var sample_mode = clip_dispatch.aa_mode;
                let max_dim = f32(max(clip_dispatch.viewport_width, clip_dispatch.viewport_height));
                let viewport_scale = max_dim / 1280.0;
                let transform_scale = max(bitcast<f32>(job.scale_hint_bits), 0.0);
                let scale_score = max(viewport_scale, transform_scale);
                if (scale_score >= 2.5) {
                    sample_mode = max(sample_mode, 3u);
                } else if (scale_score >= 1.5) {
                    sample_mode = max(sample_mode, 2u);
                }

                if (sample_mode >= 3u) {
                    a = clip_sample_8x8(job, fx, fy);
                } else if (sample_mode >= 1u) {
                    a = clip_sample_4x4(job, fx, fy);
                } else {
                    a = coarse;
                }
            }
        }
    } else if (in_fill_rect && job.fill_mode == CLIP_FILL_MODE_ROUND_RECT) {
        let fx = f32(px);
        let fy = f32(py);
        a = clip_sample_round_rect(job, fx, fy);
    }

    textureStore(
        clip_mask_out,
        vec2<i32>(i32(px), i32(py)),
        i32(job.layer),
        vec4<f32>(a, a, a, a)
    );
}
);

// ============================================================================
// Fullstack Effects WGSL Shaders
// ============================================================================

// Gaussian Blur: H + V compute shaders (separably applied to shadow/texture)
static const char* FS_EFFECTS_GAUSSIAN_BLUR_WGSL = WGSL_CODE(
struct GaussianUniforms {
    kernel_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var gaussian_input: texture_2d<f32>;
@group(0) @binding(1) var gaussian_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> gaussian_uniforms: GaussianUniforms;
@group(0) @binding(3) var<storage, read> gaussian_weights: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn gaussian_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(gaussian_input);
    let coord = vec2<i32>(gid.xy);
    if (coord.x >= i32(tex_size.x) || coord.y >= i32(tex_size.y)) { return; }
    var result = vec4<f32>(0.0);
    let radius = i32(gaussian_uniforms.kernel_size) / 2;
    for (var i: i32 = 0; i < i32(gaussian_uniforms.kernel_size); i = i + 1) {
        let offset_x = i - radius;
        let sc = clamp(coord + vec2<i32>(offset_x, 0), vec2<i32>(0), vec2<i32>(tex_size) - vec2<i32>(1));
        result = result + textureLoad(gaussian_input, sc, 0) * gaussian_weights[i];
    }
    textureStore(gaussian_output, coord, result);
}

@compute @workgroup_size(1, 256, 1)
fn gaussian_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(gaussian_input);
    let coord = vec2<i32>(gid.xy);
    if (coord.x >= i32(tex_size.x) || coord.y >= i32(tex_size.y)) { return; }
    var result = vec4<f32>(0.0);
    let radius = i32(gaussian_uniforms.kernel_size) / 2;
    for (var i: i32 = 0; i < i32(gaussian_uniforms.kernel_size); i = i + 1) {
        let offset_y = i - radius;
        let sc = clamp(coord + vec2<i32>(0, offset_y), vec2<i32>(0), vec2<i32>(tex_size) - vec2<i32>(1));
        result = result + textureLoad(gaussian_input, sc, 0) * gaussian_weights[i];
    }
    textureStore(gaussian_output, coord, result);
}
);

// Filter compute shader
static const char* FS_EFFECTS_FILTER_WGSL = WGSL_CODE(
struct FilterUniforms {
    filter_type: u32,
    param1: f32,
    param2: f32,
    param3: f32,
    param4: f32,
};

@group(0) @binding(0) var filter_src: texture_2d<f32>;
@group(0) @binding(1) var filter_dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> filter_uniforms: FilterUniforms;

fn rgb2lum(rgb: vec3<f32>) -> f32 { return dot(rgb, vec3<f32>(0.299, 0.587, 0.114)); }

fn hue_rotate(rgb: vec3<f32>, deg: f32) -> vec3<f32> {
    let a = radians(deg);
    let c = cos(a); let s = sin(a);
    let r = vec3<f32>(0.299+0.701*c+0.168*s, 0.587-0.587*c+0.330*s, 0.114-0.114*c-0.497*s);
    let g = vec3<f32>(0.299-0.299*c-0.328*s, 0.587+0.413*c+0.035*s, 0.114-0.114*c+0.292*s);
    let b = vec3<f32>(0.299-0.300*c+1.250*s, 0.587-0.588*c-1.050*s, 0.114+0.886*c-0.203*s);
    return vec3<f32>(dot(rgb,r), dot(rgb,g), dot(rgb,b));
}

fn sepia_rgb(rgb: vec3<f32>, amt: f32) -> vec3<f32> {
    let s = mat3x3<f32>(0.393,0.349,0.272, 0.769,0.686,0.534, 0.189,0.168,0.131) * rgb;
    return mix(rgb, s, amt);
}

@compute @workgroup_size(8, 8, 1)
fn filter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ts = textureDimensions(filter_src);
    let c = vec2<i32>(gid.xy);
    if (c.x >= i32(ts.x) || c.y >= i32(ts.y)) { return; }
    var col = textureLoad(filter_src, c, 0);
    var res: vec4<f32> = col;
    switch filter_uniforms.filter_type {
        case 1u: { res = vec4<f32>(col.rgb * filter_uniforms.param1, col.a); }
        case 2u: { res = vec4<f32>((col.rgb - vec3<f32>(0.5)) * filter_uniforms.param1 + vec3<f32>(0.5), col.a); }
        case 3u: { let g = mix(col.rgb, vec3<f32>(rgb2lum(col.rgb)), clamp(filter_uniforms.param1,0.0,1.0)); res = vec4<f32>(g, col.a); }
        case 4u: { res = vec4<f32>(hue_rotate(col.rgb, filter_uniforms.param1), col.a); }
        case 5u: { res = vec4<f32>(mix(col.rgb, vec3<f32>(1.0)-col.rgb, clamp(filter_uniforms.param1,0.0,1.0)), col.a); }
        case 6u: { res = vec4<f32>(col.rgb, col.a * clamp(filter_uniforms.param1,0.0,1.0)); }
        case 7u: { let lum = rgb2lum(col.rgb); res = vec4<f32>(mix(vec3<f32>(lum), col.rgb, filter_uniforms.param1), col.a); }
        case 8u: { res = vec4<f32>(sepia_rgb(col.rgb, clamp(filter_uniforms.param1,0.0,1.0)), col.a); }
        default: { res = col; }
    }
    textureStore(filter_dst, c, clamp(res, vec4<f32>(0.0), vec4<f32>(1.0)));
}
);

// Fullscreen quad vertex shader (shared by all render passes)
static const char* FS_EFFECTS_VERT_WGSL = WGSL_CODE(
@vertex fn fs_vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Overdrawn triangle technique: covers the entire clip space [-1, 1]
    // Vertices: (-1, -1), (3, -1), (-1, 3)
    var x = f32(vi & 1u) * 4.0 - 1.0; // vi=0->-1, vi=1->3, vi=2->-1
    var y = f32(vi >> 1u) * 4.0 - 1.0; // vi=0->-1, vi=1->-1, vi=2->3
    return vec4<f32>(x, y, 0.0, 1.0);
}
);

// Passthrough copy fragment shader
static const char* FS_EFFECTS_FILTER_COPY_WGSL = WGSL_CODE(
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn filter_copy_fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ts = vec2<f32>(textureDimensions(tex));
    let uv = pos.xy / ts;
    return textureSample(tex, samp, uv);
}
);

// Shadow composite: samples shadow texture at offset
static const char* FS_EFFECTS_SHADOW_COMPOSITE_WGSL = WGSL_CODE(
struct ShadowUniforms {
    color: vec4<f32>,
    offset: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var shadow_tex: texture_2d<f32>;
@group(0) @binding(1) var shadow_samp: sampler;
@group(0) @binding(2) var<uniform> shadow_uni: ShadowUniforms;
@fragment fn shadow_fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ts = vec2<f32>(textureDimensions(shadow_tex));
    let uv = pos.xy / ts;
    let off = shadow_uni.offset / ts;
    let sv = textureSample(shadow_tex, shadow_samp, uv - off);
    return shadow_uni.color * sv.a;
}
);

// Drop-shadow filter: composite blurred shadow + original content
static const char* FS_EFFECTS_DROP_SHADOW_WGSL = WGSL_CODE(
struct DSUniforms {
    color: vec4<f32>,
    offset: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var ds_blurred: texture_2d<f32>;
@group(0) @binding(1) var ds_original: texture_2d<f32>;
@group(0) @binding(2) var ds_samp: sampler;
@group(0) @binding(3) var<uniform> ds_uni: DSUniforms;
@fragment fn drop_shadow_fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ts = vec2<f32>(textureDimensions(ds_blurred));
    let uv = pos.xy / ts;
    let off = ds_uni.offset / ts;
    let blurred = textureSample(ds_blurred, ds_samp, uv - off);
    let orig = textureSample(ds_original, ds_samp, uv);
    let sc = ds_uni.color * blurred.a;
    let out_a = orig.a + sc.a * (1.0 - orig.a);
    var out_rgb: vec3<f32>;
    if (out_a > 0.0001) {
        out_rgb = (orig.rgb * orig.a + sc.rgb * sc.a * (1.0 - orig.a)) / out_a;
    } else {
        out_rgb = orig.rgb;
    }
    return vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), max(out_a, 0.0));
}
);

// Drop-shadow filter: ALL-COMPUTE pipeline
// Reads blurred shadow (src_shadow) + original scene (src_original), writes composite to dst.
// Uses 4 bindings: shadow_tex, original_tex, dst_tex, uniform_buffer.
// textureLoad for reads (no sampler needed), textureStore for write.
// This avoids WebGPU's COLOR_TARGET vs RESOURCE conflict in render passes.
static const char* FS_EFFECTS_DROP_SHADOW_C_WGSL = WGSL_CODE(
struct DSUniforms {
    color: vec4<f32>,
    offset: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var shadow_tex: texture_2d<f32>;
@group(0) @binding(1) var original_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> ds_uni: DSUniforms;

@compute @workgroup_size(8, 8, 1)
fn drop_shadow_c_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ts = vec2<i32>(textureDimensions(shadow_tex));
    let c = gid.xy;
    if (c.x >= u32(ts.x) || c.y >= u32(ts.y)) { return; }
    let ts_f = vec2<f32>(ts);
    let off = vec2<i32>(ds_uni.offset);
    let blurred = textureLoad(shadow_tex, vec2<i32>(c) - off, 0);
    let orig = textureLoad(original_tex, vec2<i32>(c), 0);
    let sc = ds_uni.color * blurred.a;
    let out_a = orig.a + sc.a * (1.0 - orig.a);
    var out_rgb: vec3<f32>;
    if (out_a > 0.0001) {
        out_rgb = (orig.rgb * orig.a + sc.rgb * sc.a * (1.0 - orig.a)) / out_a;
    } else {
        out_rgb = orig.rgb;
    }
    textureStore(dst_tex, vec2<i32>(c), vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), max(out_a, 0.0)));
}
);

#endif
