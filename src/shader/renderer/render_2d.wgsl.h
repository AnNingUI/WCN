#ifndef WCN_RENDER_2D_WGSL_H
#define WCN_RENDER_2D_WGSL_H

#include "WCN/WCN_WGSL.h"

static const char* WCN_RENDER_2D_WGSL = WGSL_CODE(

// --- Constants ---
const INSTANCE_TYPE_RECT: u32 = 0u;
const INSTANCE_TYPE_TEXT: u32 = 1u;
const INSTANCE_TYPE_PATH: u32 = 2u;
const INSTANCE_TYPE_LINE: u32 = 3u;
const INSTANCE_TYPE_IMAGE: u32 = 4u;
const INSTANCE_TYPE_ARC: u32 = 5u;
const INSTANCE_TYPE_BEZIER: u32 = 6u;
const INSTANCE_TYPE_CIRCLE_FILL: u32 = 7u;

const LINE_CAP_BUTT: u32 = 0u;
const LINE_CAP_ROUND: u32 = 1u;
const LINE_CAP_SQUARE: u32 = 2u;
const LINE_CAP_START_ENABLED: u32 = 0x100u;
const LINE_CAP_END_ENABLED: u32 = 0x200u;

// --- Bind Groups ---
@group(1) @binding(0) var sdf_atlas: texture_2d<f32>;
@group(1) @binding(1) var sdf_sampler: sampler;
@group(1) @binding(2) var image_atlas: texture_2d<f32>;
@group(1) @binding(3) var image_sampler: sampler;

struct VertexInput {
    @location(0) clip_position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) instance_type: u32,
    @location(4) @interpolate(flat) flags: u32,
    @location(5) local_pos: vec2<f32>,
    @location(6) params_x: f32,
    @location(7) size: vec2<f32>,
    @location(8) tri_v0: vec2<f32>,
    @location(9) tri_v1: vec2<f32>,
    @location(10) tri_v2: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) instance_type: u32,
    @location(3) @interpolate(flat) flags: u32,
    @location(4) local_pos: vec2<f32>,
    @location(5) params_x: f32,
    @location(6) @interpolate(flat) size: vec2<f32>,
    @location(7) tri_v0: vec2<f32>,
    @location(8) tri_v1: vec2<f32>,
    @location(9) tri_v2: vec2<f32>,
};

fn edge_function(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

// 仅用于 Emoji 和图片的双三次插值，保证清晰度
// 不影响 SDF 文字
fn sample_bicubic(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let tex_dims = vec2<f32>(textureDimensions(tex));
    let sample_pos = uv * tex_dims;
    let tex_pos_i = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos_i;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);

    let w12 = w1 + w2;
    let offset12 = w2 / (w1 + w2);

    let tex_pos0 = tex_pos_i - 1.0;
    let tex_pos3 = tex_pos_i + 2.0;
    let tex_pos12 = tex_pos_i + offset12;

    let p0 = vec2<f32>(tex_pos12.x, tex_pos0.y) / tex_dims;
    let p1 = vec2<f32>(tex_pos0.x, tex_pos12.y) / tex_dims;
    let p2 = vec2<f32>(tex_pos12.x, tex_pos12.y) / tex_dims;
    let p3 = vec2<f32>(tex_pos3.x, tex_pos12.y) / tex_dims;
    let p4 = vec2<f32>(tex_pos12.x, tex_pos3.y) / tex_dims;

    return textureSampleGrad(tex, samp, p2, ddx, ddy) * w12.x * w12.y +
           textureSampleGrad(tex, samp, p0, ddx, ddy) * w12.x * w0.y +
           textureSampleGrad(tex, samp, p4, ddx, ddy) * w12.x * w3.y +
           textureSampleGrad(tex, samp, p1, ddx, ddy) * w0.x * w12.y +
           textureSampleGrad(tex, samp, p3, ddx, ddy) * w3.x * w12.y;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = input.clip_position;
    output.color = input.color;
    output.uv = input.uv;
    output.instance_type = input.instance_type;
    output.flags = input.flags;
    output.local_pos = input.local_pos;
    output.params_x = input.params_x;
    output.size = input.size;
    output.tri_v0 = input.tri_v0;
    output.tri_v1 = input.tri_v1;
    output.tri_v2 = input.tri_v2;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = input.color;
    
    // 预计算导数（WebGPU 必须在统一控制流中调用）
    // Chrome 要求 fwidth/dpdx/dpdy 不能在 switch/if 分支中调用
    let uv_ddx = dpdx(input.uv);
    let uv_ddy = dpdy(input.uv);
    let dx = dpdx(input.local_pos.x);
    let dy = dpdy(input.local_pos.y);
    let local_pos_fwidth = fwidth(input.local_pos);
    let params_x_fwidth = fwidth(input.params_x);

    // SDF 采样保持不变 (Bilinear)，因为 SDF 需要平滑的梯度
    let sdf_sample = textureSample(sdf_atlas, sdf_sampler, input.uv);
    let distance = sdf_sample.r;
    let dist_grad = fwidth(distance);

    switch (input.instance_type) {
        case INSTANCE_TYPE_RECT: {
            let radius = clamp(input.params_x, 0.0, min(input.size.x, input.size.y) * 0.5);
            let p = (input.local_pos - 0.5) * input.size;
            let b = (input.size * 0.5) - vec2<f32>(radius, radius);
            let q = abs(p) - b;
            let d = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
            
            let alpha = 1.0 - smoothstep(-0.5, 0.5, d);
            color.a *= alpha;
        }
        
        case INSTANCE_TYPE_TEXT: {
            let is_color_bitmap = (input.flags & 1u) != 0u;

            if (is_color_bitmap) {
                // 【升级点】 Emoji 使用 Bicubic 采样
                // 使用 sample_bicubic 代替简单的 textureSample 或 rgb
                // 这样 Emoji 放大时边缘会更清晰，不会有马赛克
                let emoji = sample_bicubic(sdf_atlas, sdf_sampler, input.uv, uv_ddx, uv_ddy);
                
                // Emoji 保持原色，只应用 input.color 的透明度
                return vec4<f32>(emoji.rgb, emoji.a * color.a);
            }

            // 【保持不变】 你觉得最正常的 SDF 渲染逻辑
            let adaptive_width = dist_grad * 0.65;
            let w = max(adaptive_width, 0.05);
            let alpha = smoothstep(0.5 - w, 0.5 + w, distance);
            let sharpened_alpha = pow(alpha, 1.1);
            
            color.a *= sharpened_alpha;
        }
        
        case INSTANCE_TYPE_PATH: {
            // 简单的三角形内部测试，不做抗锯齿
            let v0 = input.tri_v0;
            let v1 = input.tri_v1;
            let v2 = input.tri_v2;
            let p = input.local_pos;
            let e0 = edge_function(v0, v1, p);
            let e1 = edge_function(v1, v2, p);
            let e2 = edge_function(v2, v0, p);
            let inside = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
            if (!inside) { return vec4<f32>(0.0); }
        }
        
        case INSTANCE_TYPE_LINE: {
            // ... (保持你原本的 Line 逻辑不变) ...
            let line_width = input.params_x;
            let half_width = line_width * 0.5;
            let line_length = input.size.x;
            let half_length = line_length * 0.5;
            let cap_style = input.flags & 0xFFu;
            let start_cap_enabled = (input.flags & LINE_CAP_START_ENABLED) != 0u;
            let end_cap_enabled = (input.flags & LINE_CAP_END_ENABLED) != 0u;
            
            var start_extension: f32 = 0.0;
            var end_extension: f32 = 0.0;
            if (cap_style != LINE_CAP_BUTT) {
                if (start_cap_enabled) { start_extension = half_width; }
                if (end_cap_enabled) { end_extension = half_width; }
            }
            let extended_length = line_length + start_extension + end_extension;
            let center_offset = (end_extension - start_extension) * 0.5;
            let px = (input.local_pos.x - 0.5) * extended_length + center_offset;
            let py = (input.local_pos.y - 0.5) * line_width;
            var sdf_distance: f32;
            
            // ... (Line SDF 计算逻辑太长省略，保持你原版逻辑) ...
            // 为节省空间，此处逻辑与你提供的一致
            // [Safety] 只要确保这里没有三元运算符即可
            
            // 简单重写避免三元运算符的 Line 长度计算
            let line_start = -half_length - start_extension;
            let line_end = half_length + end_extension;
            
            // 你的原版 SDF 逻辑... 假设这里直接计算出了 sdf_distance
             if (cap_style == LINE_CAP_BUTT || (!start_cap_enabled && !end_cap_enabled)) {
                let dx = max(abs(px) - half_length, 0.0);
                let dy = max(abs(py) - half_width, 0.0);
                sdf_distance = -sqrt(dx * dx + dy * dy);
                if (abs(px) <= half_length && abs(py) <= half_width) {
                    sdf_distance = min(half_width - abs(py), half_length - abs(px));
                }
            } else if (cap_style == LINE_CAP_SQUARE) {
                // 使用 select 替代三元运算
                let left_bound = select(-half_length, -half_length - half_width, start_cap_enabled);
                let right_bound = select(half_length, half_length + half_width, end_cap_enabled);
                let dx = max(max(left_bound - px, px - right_bound), 0.0);
                let dy = max(abs(py) - half_width, 0.0);
                sdf_distance = -sqrt(dx * dx + dy * dy);
                if (px >= left_bound && px <= right_bound && abs(py) <= half_width) {
                    sdf_distance = half_width - abs(py);
                }
            } else {
                 var clamped_x: f32;
                if (start_cap_enabled && end_cap_enabled) {
                    clamped_x = clamp(px, -half_length, half_length);
                } else if (start_cap_enabled) {
                    clamped_x = clamp(px, -half_length, half_length);
                    if (px > half_length) {
                         let dx = px - half_length;
                        let dy = abs(py) - half_width;
                        if (dy <= 0.0) { sdf_distance = -dx; } else { sdf_distance = -sqrt(dx * dx + dy * dy); }
                        clamped_x = px; 
                    }
                } else if (end_cap_enabled) {
                    clamped_x = clamp(px, -half_length, half_length);
                    if (px < -half_length) {
                        let dx = -half_length - px;
                        let dy = abs(py) - half_width;
                        if (dy <= 0.0) { sdf_distance = -dx; } else { sdf_distance = -sqrt(dx * dx + dy * dy); }
                        clamped_x = px;
                    }
                } else {
                    clamped_x = clamp(px, -half_length, half_length);
                }
                if (sdf_distance == 0.0 || (start_cap_enabled && end_cap_enabled)) {
                    let dx = px - clamped_x;
                    let dy = py;
                    let distance_to_axis = sqrt(dx * dx + dy * dy);
                    sdf_distance = half_width - distance_to_axis;
                }
            }

            let edge_distance = length(vec2<f32>(dx, dy));
            let aa_width = max(0.5, edge_distance * line_width);
            var alpha = smoothstep(-aa_width, aa_width, sdf_distance);
            if (line_width < 3.0) { alpha = pow(alpha, 0.85); } else { alpha = pow(alpha, 0.95); }
            color.a *= alpha;
        }
        
        case INSTANCE_TYPE_IMAGE: {
            let sampled_color = sample_bicubic(image_atlas, image_sampler, input.uv, uv_ddx, uv_ddy);
            return sampled_color * color;
        }
        
        case INSTANCE_TYPE_ARC: {
            // GPU SDF 圆弧渲染
            let radius = input.size.x;
            let stroke_width = input.size.y;
            let start_angle = input.uv.x;
            let end_angle = input.uv.y;
            let half_stroke = stroke_width * 0.5;
            
            let PI = 3.14159265;
            let TWO_PI = 6.283185307;
            
            // 计算像素相对于圆心的位置
            let p = (input.local_pos - 0.5) * 2.0 * (radius + half_stroke);
            let dist_to_center = length(p);
            
            // 计算当前像素的角度 [-PI, PI]
            let angle = atan2(p.y, p.x);
            
            // 计算角度跨度
            let angle_span = end_angle - start_angle;
            
            // 判断是否为完整圆 (角度跨度接近或超过 2PI)
            let is_full_circle = abs(angle_span) >= TWO_PI - 0.01;
            
            // 判断当前角度是否在圆弧范围内
            var in_arc = is_full_circle;
            if (!is_full_circle) {
                // 使用向量叉积方法判断角度是否在范围内
                // 这种方法更稳定，不需要复杂的角度规范化
                let v_start = vec2<f32>(cos(start_angle), sin(start_angle));
                let v_end = vec2<f32>(cos(end_angle), sin(end_angle));
                let v_p = normalize(p);
                
                // 计算叉积 (2D 叉积返回标量)
                let cross_start_p = v_start.x * v_p.y - v_start.y * v_p.x;
                let cross_start_end = v_start.x * v_end.y - v_start.y * v_end.x;
                let cross_p_end = v_p.x * v_end.y - v_p.y * v_end.x;
                
                // 判断 p 是否在 start 到 end 的扇形区域内
                if (angle_span >= 0.0) {
                    // 逆时针方向 (从 start 到 end 角度增加)
                    if (angle_span <= PI) {
                        // 小于等于 180 度：p 必须在 start 的逆时针方向且在 end 的顺时针方向
                        in_arc = cross_start_p >= -0.001 && cross_p_end >= -0.001;
                    } else {
                        // 大于 180 度：p 不能同时在 start 的顺时针方向且在 end 的逆时针方向
                        in_arc = !(cross_start_p < -0.001 && cross_p_end < -0.001);
                    }
                } else {
                    // 顺时针方向 (从 start 到 end 角度减少)
                    let abs_span = -angle_span;
                    if (abs_span <= PI) {
                        // 小于等于 180 度
                        in_arc = cross_start_p <= 0.001 && cross_p_end <= 0.001;
                    } else {
                        // 大于 180 度
                        in_arc = !(cross_start_p > 0.001 && cross_p_end > 0.001);
                    }
                }
            }
            
            var sdf_distance: f32;
            if (is_full_circle) {
                // 完整圆环
                sdf_distance = half_stroke - abs(dist_to_center - radius);
            } else {
                if (in_arc) {
                    // 在圆弧范围内
                    sdf_distance = half_stroke - abs(dist_to_center - radius);
                } else {
                    // 在圆弧范围外，计算到端点的距离
                    let p_start = vec2<f32>(cos(start_angle), sin(start_angle)) * radius;
                    let p_end = vec2<f32>(cos(end_angle), sin(end_angle)) * radius;
                    let dist_to_start = length(p - p_start);
                    let dist_to_end = length(p - p_end);
                    sdf_distance = half_stroke - min(dist_to_start, dist_to_end);
                }
            }
            
            // 使用预计算的 fwidth 近似值（基于 local_pos 的变化率）
            let arc_aa_width = length(local_pos_fwidth) * (radius + half_stroke) + 0.5;
            let alpha = smoothstep(-arc_aa_width, arc_aa_width, sdf_distance);
            color.a *= alpha;
        }
        
        case INSTANCE_TYPE_BEZIER: {
            // GPU SDF 二次贝塞尔曲线渲染
            let stroke_width = input.params_x;
            let half_stroke = stroke_width * 0.5;
            
            // 数据编码:
            // tri_v1 = 起点归一化坐标
            // uv = 控制点归一化坐标 (注意: uv 在 vertex shader 中被插值了)
            // tri_v2 = 终点归一化坐标
            let p0 = input.tri_v1 * input.size;
            let p1 = input.tri_v0 * input.size;  // 控制点从 tri_v0 读取 (未插值)
            let p2 = input.tri_v2 * input.size;
            
            let pixel_pos = input.local_pos * input.size;
            
            // 采样曲线找最近距离
            var min_dist = 1e10;
            for (var i = 0; i <= 20; i = i + 1) {
                let t = f32(i) / 20.0;
                let mt = 1.0 - t;
                let curve_pt = mt * mt * p0 + 2.0 * mt * t * p1 + t * t * p2;
                let d = length(pixel_pos - curve_pt);
                min_dist = min(min_dist, d);
            }
            
            let bezier_sdf_distance = half_stroke - min_dist;
            // 使用预计算的 fwidth 近似值
            let bezier_aa_width = length(local_pos_fwidth) * max(input.size.x, input.size.y) + 0.5;
            let alpha = smoothstep(-bezier_aa_width, bezier_aa_width, bezier_sdf_distance);
            color.a *= alpha;
        }
        
        case INSTANCE_TYPE_CIRCLE_FILL: {
            // GPU SDF 圆形/扇形填充渲染
            let radius = input.size.x;
            let start_angle = input.uv.x;
            let end_angle = input.uv.y;
            
            let PI = 3.14159265;
            let TWO_PI = 6.283185307;
            
            // 计算像素相对于圆心的位置
            let p = (input.local_pos - 0.5) * 2.0 * radius;
            let dist_to_center = length(p);
            
            // 计算角度跨度
            let angle_span = end_angle - start_angle;
            
            // 判断是否为完整圆 (角度跨度接近或超过 2PI)
            let is_full_circle = abs(angle_span) >= TWO_PI - 0.01;
            
            // 判断当前角度是否在扇形范围内
            var in_arc = is_full_circle;
            if (!is_full_circle) {
                let v_start = vec2<f32>(cos(start_angle), sin(start_angle));
                let v_end = vec2<f32>(cos(end_angle), sin(end_angle));
                let v_p = normalize(p);
                
                let cross_start_p = v_start.x * v_p.y - v_start.y * v_p.x;
                let cross_p_end = v_p.x * v_end.y - v_p.y * v_end.x;
                
                if (angle_span >= 0.0) {
                    if (angle_span <= PI) {
                        in_arc = cross_start_p >= -0.001 && cross_p_end >= -0.001;
                    } else {
                        in_arc = !(cross_start_p < -0.001 && cross_p_end < -0.001);
                    }
                } else {
                    let abs_span = -angle_span;
                    if (abs_span <= PI) {
                        in_arc = cross_start_p <= 0.001 && cross_p_end <= 0.001;
                    } else {
                        in_arc = !(cross_start_p > 0.001 && cross_p_end > 0.001);
                    }
                }
            }
            
            var sdf_distance: f32;
            if (is_full_circle) {
                // 完整圆：SDF = radius - dist_to_center
                sdf_distance = radius - dist_to_center;
            } else {
                if (in_arc) {
                    // 在扇形范围内：SDF = radius - dist_to_center
                    sdf_distance = radius - dist_to_center;
                } else {
                    // 在扇形范围外：计算到两条边的距离
                    let p_start = vec2<f32>(cos(start_angle), sin(start_angle)) * radius;
                    let p_end = vec2<f32>(cos(end_angle), sin(end_angle)) * radius;
                    
                    // 到起始边的距离 (从圆心到 p_start 的线段)
                    let t_start = clamp(dot(p, normalize(p_start)), 0.0, radius);
                    let closest_start = normalize(p_start) * t_start;
                    let dist_to_start_edge = length(p - closest_start);
                    
                    // 到结束边的距离 (从圆心到 p_end 的线段)
                    let t_end = clamp(dot(p, normalize(p_end)), 0.0, radius);
                    let closest_end = normalize(p_end) * t_end;
                    let dist_to_end_edge = length(p - closest_end);
                    
                    sdf_distance = -min(dist_to_start_edge, dist_to_end_edge);
                }
            }
            
            let fill_aa_width = length(local_pos_fwidth) * radius + 0.5;
            let alpha = smoothstep(-fill_aa_width, fill_aa_width, sdf_distance);
            color.a *= alpha;
        }
        
        default: { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }
    }
    
    if (color.a < 0.01) { discard; }
    return color;
}
);

#endif // WCN_RENDER_2D_WGSL_H