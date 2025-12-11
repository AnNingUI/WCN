#ifndef WCN_RENDER_2D_WGSL_H
#define WCN_RENDER_2D_WGSL_H

#include "WCN/WCN_WGSL.h"

// Unified rendering shader - single pipeline for all primitive types
static const char* WCN_RENDER_2D_WGSL = WGSL_CODE(

// Instance types
const INSTANCE_TYPE_RECT: u32 = 0u;
const INSTANCE_TYPE_TEXT: u32 = 1u;
const INSTANCE_TYPE_PATH: u32 = 2u;
const INSTANCE_TYPE_LINE: u32 = 3u;
const INSTANCE_TYPE_IMAGE: u32 = 4u;

// Line cap styles (stored in flags field, bits 0-7)
const LINE_CAP_BUTT: u32 = 0u;    // Flat cap (no extension)
const LINE_CAP_ROUND: u32 = 1u;   // Round cap (circular)
const LINE_CAP_SQUARE: u32 = 2u;  // Square cap (rectangular extension)

// Line cap enable flags (bits 8-9)
const LINE_CAP_START_ENABLED: u32 = 0x100u;  // Bit 8: render start cap
const LINE_CAP_END_ENABLED: u32 = 0x200u;    // Bit 9: render end cap

// Instance structure definition kept for bind group compatibility
// SDF Atlas (for text rendering)
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
    @location(4) local_pos: vec2<f32>,  // Local quad position for AA
    @location(5) params_x: f32,         // Type-specific parameter
    @location(6) @interpolate(flat) size: vec2<f32>,  // Instance size (for LINE: length, width)
    @location(7) tri_v0: vec2<f32>,     // Triangle vertex 0 (for PATH instances)
    @location(8) tri_v1: vec2<f32>,     // Triangle vertex 1 (for PATH instances)
    @location(9) tri_v2: vec2<f32>,     // Triangle vertex 2 (for PATH instances)
};

// Vertex shader now consumes pre-expanded vertex data
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

// Fragment shader - branches based on instance type
fn edge_function(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = input.color;
    
    // Pre-compute derivatives in uniform control flow (required by WebGPU)
    // These must be calculated before any branching
    let uv_ddx = dpdx(input.uv);
    let uv_ddy = dpdy(input.uv);
    let sdf_sample = textureSample(sdf_atlas, sdf_sampler, input.uv);
    let distance = sdf_sample.r;
    let dist_grad = fwidth(distance);
    let dx = dpdx(input.local_pos.x);
    let dy = dpdy(input.local_pos.y);
    
    // Branch based on instance type
    switch (input.instance_type) {
        case INSTANCE_TYPE_RECT: {
            // SDF-based Rounded Rectangle
            // params_x holds the corner radius
            let radius = clamp(input.params_x, 0.0, min(input.size.x, input.size.y) * 0.5);
            
            // Convert local_pos (0..1) to centered coordinates
            let p = (input.local_pos - 0.5) * input.size;
            let b = (input.size * 0.5) - vec2<f32>(radius, radius);
            
            // SDF for rounded box
            let q = abs(p) - b;
            let d = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
            
            // Anti-aliasing
            // We can use the pre-computed derivatives dx/dy for the gradient length
            // or simply use a fixed width (1.0 pixel) since d is in pixel units?
            // Wait, p is in logical pixels (world space local).
            // We need screen space width.
            // length(vec2(dx, dy)) gives the rate of change of local_pos.
            // size * length(vec2(dx, dy)) gives rate of change in pixels.
            // Simpler: fwidth(d). 
            // Since we are in non-uniform control flow, we can use the pre-computed dx/dy
            // to estimate the gradient.
            // Gradient of SDF is unit length in world space? 
            // If the transformation is uniform scale, yes. 
            // If non-uniform, it's harder.
            // Let's assume uniform-ish scale or use fwidth(d) with caution.
            // NOTE: Using fwidth here might be flagged by strict validators.
            // But since 'd' depends on 'p' which depends on 'local_pos', 
            // and we have dx/dy for local_pos...
            
            // Let's use a safe 1.0 pixel AA width assuming 1:1 screen mapping for now,
            // or better: use the 'dist_grad' if it was computed from screen space?
            // No, 'dist_grad' was for text SDF.
            
            // Standard derivative approximation using pre-computed dx, dy
            // Gradient of p.x is input.size.x * dx
            // Gradient of p.y is input.size.y * dy
            // Approximate pixel size in local space:
            let px_size = length(vec2<f32>(dx * input.size.x, dy * input.size.y));
            // Or just fwidth of d if available/safe.
            
            // Smoothstep for AA
            let alpha = 1.0 - smoothstep(-0.5, 0.5, d); // 1px smooth edge
            
            color.a *= alpha;
            if (color.a < 0.01) {
                return vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }
            return color;
        }
        case INSTANCE_TYPE_TEXT: {
            // ... (existing text code)
            // New "Alpha + Offsets" SDF Rendering
            // Texture Format:
            // R: Signed Distance (Normalized, 0.5 = Edge)
            // G: Subpixel Offset X (Vector to nearest edge)
            // B: Subpixel Offset Y (Vector to nearest edge)
            // A: Opacity (usually 1.0)
            
            // We primarily use the Red channel (Distance) which has been generated 
            // with subpixel precision using the ESDT algorithm.
            // Note: 'distance' and 'dist_grad' (fwidth) are pre-calculated 
            // outside this non-uniform control flow block for portability.
            
            // AA Width Calculation
            // The SDF spread is approx 3 pixels in the atlas.
            // adaptive_width determines the smoothness of the edge.
            // Lower 'sharpness' constant allows for crisper edges on high-DPI screens.
            let sharpness = 0.1; 
            let adaptive_width = max(sharpness, dist_grad * 0.7);
            
            // Apply Smoothstep
            // Maps the distance field to opacity with an anti-aliased edge
            let alpha = smoothstep(0.5 - adaptive_width, 0.5 + adaptive_width, distance);
            
            // Gamma / Perceptual Correction
            // Slight sharpening curve to enhance readability
            let sharpened_alpha = pow(alpha, 1.1);
            
            color.a *= sharpened_alpha;
            
            // Optimization: Return transparent for invisible pixels
            // (Avoid 'discard' for better compatibility)
            if (color.a < 0.01) {
                return vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }
            
            return color;
        }
        case INSTANCE_TYPE_PATH: {
            let v0 = input.tri_v0;
            let v1 = input.tri_v1;
            let v2 = input.tri_v2;
            let p = input.local_pos;

            let e0 = edge_function(v0, v1, p);
            let e1 = edge_function(v1, v2, p);
            let e2 = edge_function(v2, v0, p);

            // Support both winding orders (CCW and CW)
            // CCW: All edges >= 0
            // CW:  All edges <= 0
            let has_pos = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0);
            let has_neg = (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
            
            if (!(has_pos || has_neg)) {
                return vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }
            
            return color;
        }
        case INSTANCE_TYPE_LINE: {
            // SDF-based line rendering with multiple cap styles
            // Reference: https://iquilezles.org/articles/distfunctions2d/
            
            let line_width = input.params_x;
            let half_width = line_width * 0.5;
            let line_length = input.size.x;
            let half_length = line_length * 0.5;
            
            // Get line cap style and enable flags
            let cap_style = input.flags & 0xFFu;
            let start_cap_enabled = (input.flags & LINE_CAP_START_ENABLED) != 0u;
            let end_cap_enabled = (input.flags & LINE_CAP_END_ENABLED) != 0u;
            
            // Calculate extensions based on cap style and which caps are enabled
            var start_extension: f32 = 0.0;
            var end_extension: f32 = 0.0;
            
            if (cap_style != LINE_CAP_BUTT) {
                if (start_cap_enabled) {
                    start_extension = half_width;
                }
                if (end_cap_enabled) {
                    end_extension = half_width;
                }
            }
            
            let extended_length = line_length + start_extension + end_extension;
            let center_offset = (end_extension - start_extension) * 0.5;
            
            // Convert local_pos [0,1] to pixel coordinates centered at origin
            let px = (input.local_pos.x - 0.5) * extended_length + center_offset;
            let py = (input.local_pos.y - 0.5) * line_width;
            
            // Calculate SDF distance based on cap style and enabled caps
            var sdf_distance: f32;
            
            // Calculate the line segment bounds (considering which caps are enabled)
            let line_start = -half_length - start_extension;
            let line_end = half_length + end_extension;
            
            if (cap_style == LINE_CAP_BUTT || (!start_cap_enabled && !end_cap_enabled)) {
                // BUTT or no caps: Simple rectangle
                let dx = max(abs(px) - half_length, 0.0);
                let dy = max(abs(py) - half_width, 0.0);
                sdf_distance = -sqrt(dx * dx + dy * dy);
                if (abs(px) <= half_length && abs(py) <= half_width) {
                    sdf_distance = min(half_width - abs(py), half_length - abs(px));
                }
            } else if (cap_style == LINE_CAP_SQUARE) {
                // SQUARE: Rectangle with square extensions where enabled
                let left_bound = select(-half_length, -half_length - half_width, start_cap_enabled);
                let right_bound = select(half_length, half_length + half_width, end_cap_enabled);
                
                let dx = max(max(left_bound - px, px - right_bound), 0.0);
                let dy = max(abs(py) - half_width, 0.0);
                sdf_distance = -sqrt(dx * dx + dy * dy);
                if (px >= left_bound && px <= right_bound && abs(py) <= half_width) {
                    sdf_distance = half_width - abs(py);
                }
            } else {
                // ROUND: Capsule with selective caps
                // Clamp to line segment, but extend where caps are enabled
                var clamped_x: f32;
                if (start_cap_enabled && end_cap_enabled) {
                    // Both caps: full capsule
                    clamped_x = clamp(px, -half_length, half_length);
                } else if (start_cap_enabled) {
                    // Only start cap: round start, flat end
                    clamped_x = clamp(px, -half_length, half_length);
                    if (px > half_length) {
                        // Beyond end: use rectangle distance
                        let dx = px - half_length;
                        let dy = abs(py) - half_width;
                        if (dy <= 0.0) {
                            sdf_distance = -dx;
                        } else {
                            sdf_distance = -sqrt(dx * dx + dy * dy);
                        }
                        // Skip capsule calculation
                        clamped_x = px;  // Force no rounding at end
                    }
                } else if (end_cap_enabled) {
                    // Only end cap: flat start, round end
                    clamped_x = clamp(px, -half_length, half_length);
                    if (px < -half_length) {
                        // Beyond start: use rectangle distance
                        let dx = -half_length - px;
                        let dy = abs(py) - half_width;
                        if (dy <= 0.0) {
                            sdf_distance = -dx;
                        } else {
                            sdf_distance = -sqrt(dx * dx + dy * dy);
                        }
                        // Skip capsule calculation
                        clamped_x = px;  // Force no rounding at start
                    }
                } else {
                    // No caps: rectangle
                    clamped_x = clamp(px, -half_length, half_length);
                }
                
                // Only calculate capsule distance if we didn't already set it above
                if (sdf_distance == 0.0 || (start_cap_enabled && end_cap_enabled)) {
                    let dx = px - clamped_x;
                    let dy = py;
                    let distance_to_axis = sqrt(dx * dx + dy * dy);
                    sdf_distance = half_width - distance_to_axis;
                }
            }
            
            // Adaptive anti-aliasing with sharpening
            // Use pre-computed derivatives instead of fwidth(sdf_distance)
            let edge_distance = length(vec2<f32>(dx, dy));
            let aa_width = max(0.5, edge_distance * line_width);
            
            // Apply smoothstep
            var alpha = smoothstep(-aa_width, aa_width, sdf_distance);
            
            // Apply sharpening curve for crisper edges
            // This makes thin lines look sharper
            if (line_width < 3.0) {
                alpha = pow(alpha, 0.85);  // More sharpening for thin lines
            } else {
                alpha = pow(alpha, 0.95);  // Subtle sharpening for thick lines
            }
            
            color.a *= alpha;
            
            // Discard fully transparent fragments
            if (color.a < 0.01) {
                return vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }
            
            return color;
        }
        case INSTANCE_TYPE_IMAGE: {
            // Move texture sampling to uniform control flow
            let sampled_color = textureSampleGrad(
                image_atlas, 
                image_sampler, 
                input.uv, 
                uv_ddx, 
                uv_ddy
            );
            
            return sampled_color * color;
        }
        default: {
            // Unknown type - return magenta for debugging
            return vec4<f32>(1.0, 0.0, 1.0, 1.0);
        }
    }
});

#endif // WCN_RENDER_2D_WGSL_H
