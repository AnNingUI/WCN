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

// Line cap styles (stored in flags field, bits 0-7)
const LINE_CAP_BUTT: u32 = 0u;    // Flat cap (no extension)
const LINE_CAP_ROUND: u32 = 1u;   // Round cap (circular)
const LINE_CAP_SQUARE: u32 = 2u;  // Square cap (rectangular extension)

// Line cap enable flags (bits 8-9)
const LINE_CAP_START_ENABLED: u32 = 0x100u;  // Bit 8: render start cap
const LINE_CAP_END_ENABLED: u32 = 0x200u;    // Bit 9: render end cap

// Instance structure (64 bytes, GPU-aligned)
// Carefully laid out to achieve exactly 64 bytes with GPU alignment rules
struct Instance {
    position: vec2<f32>,        // 8 bytes: (x, y) in screen space
    size: vec2<f32>,            // 8 bytes: (width, height)
    uv: vec2<f32>,              // 8 bytes: UV start coordinates
    uvSize: vec2<f32>,          // 8 bytes: UV size (not max!)
    transform: vec4<f32>,       // 16 bytes: 2x2 matrix as vec4 (m00, m01, m10, m11)
    color: u32,                 // 4 bytes: packed RGBA (0xAABBGGRR)
    instance_type: u32,         // 4 bytes: instance type
    flags: u32,                 // 4 bytes: rendering flags
    params_x: f32,              // 4 bytes: type-specific parameter 1
};                              // Total: 64 bytes

// Uniforms
struct Uniforms {
    viewport_size: vec2<f32>,
    padding: vec2<f32>,
};

// Bind groups
@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

// SDF Atlas (for text rendering)
@group(1) @binding(0) var sdf_atlas: texture_2d<f32>;
@group(1) @binding(1) var sdf_sampler: sampler;

// Vertex output
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

// Unpack u32 color to vec4<f32>
// Format: 0xAABBGGRR (alpha, blue, green, red)
fn unpack_color(packed: u32) -> vec4<f32> {
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

// Vertex shader - generates quad from instance data
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var output: VertexOutput;
    
    // Load instance data
    let instance = instances[instance_index];
    
    // Generate quad vertices (0,0) to (1,1) in local space
    // Vertex order: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1), 4=(0,1), 5=(1,0)
    var local_pos: vec2<f32>;
    switch (vertex_index) {
        case 0u: { local_pos = vec2<f32>(0.0, 0.0); }
        case 1u: { local_pos = vec2<f32>(1.0, 0.0); }
        case 2u: { local_pos = vec2<f32>(0.0, 1.0); }
        case 3u: { local_pos = vec2<f32>(1.0, 1.0); }
        case 4u: { local_pos = vec2<f32>(0.0, 1.0); }
        default: { local_pos = vec2<f32>(1.0, 0.0); }
    }
    
    // Special handling for LINE instances
    var sized_pos: vec2<f32>;
    if (instance.instance_type == INSTANCE_TYPE_LINE) {
        // For lines, uv stores the normalized direction vector
        let dir_x = instance.uv.x;
        let dir_y = instance.uv.y;
        
        // Create perpendicular vector for line width
        let perp_x = -dir_y;
        let perp_y = dir_x;
        
        // Line length and width from size
        let length = instance.size.x;
        let width = instance.size.y;
        let half_width = width * 0.5;
        
        // Get line cap style and enable flags
        let cap_style = instance.flags & 0xFFu;  // Bits 0-7
        let start_cap_enabled = (instance.flags & LINE_CAP_START_ENABLED) != 0u;
        let end_cap_enabled = (instance.flags & LINE_CAP_END_ENABLED) != 0u;
        
        // Calculate extension based on cap style and which caps are enabled
        var start_extension: f32 = 0.0;
        var end_extension: f32 = 0.0;
        
        if (cap_style != LINE_CAP_BUTT) {
            // SQUARE and ROUND caps extend by half_width
            if (start_cap_enabled) {
                start_extension = half_width;
            }
            if (end_cap_enabled) {
                end_extension = half_width;
            }
        }
        
        let extended_length = length + start_extension + end_extension;
        
        // Map local_pos (0-1) to extended line coordinates
        // Adjust center offset based on asymmetric extensions
        let center_offset = (end_extension - start_extension) * 0.5;
        let along = (local_pos.x - 0.5) * extended_length + center_offset;
        let across = (local_pos.y - 0.5) * width;
        
        // Position along line direction + across perpendicular
        sized_pos = vec2<f32>(
            along * dir_x + across * perp_x,
            along * dir_y + across * perp_y
        );
    } else {
        // Standard quad for other instance types
        sized_pos = local_pos * instance.size;
    }
    
    // Apply 2x2 transform matrix
    // Matrix stored as vec4: (m00, m01, m10, m11)
    let m00 = instance.transform.x;
    let m01 = instance.transform.y;
    let m10 = instance.transform.z;
    let m11 = instance.transform.w;
    
    let transformed_x = sized_pos.x * m00 + sized_pos.y * m01;
    let transformed_y = sized_pos.x * m10 + sized_pos.y * m11;
    let transformed_pos = vec2<f32>(transformed_x, transformed_y);
    
    // Apply position offset
    let world_pos = transformed_pos + instance.position;
    
    // Convert to NDC coordinates
    let ndc_x = (world_pos.x / uniforms.viewport_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world_pos.y / uniforms.viewport_size.y) * 2.0;
    
    output.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    output.color = unpack_color(instance.color);
    
    // Interpolate UV coordinates
    // uv is the min corner, uvSize is the size (not max!)
    output.uv = instance.uv + local_pos * instance.uvSize;
    
    output.instance_type = instance.instance_type;
    output.flags = instance.flags;
    output.local_pos = local_pos;
    output.params_x = instance.params_x;
    output.size = instance.size;
    
    // For PATH instances, pass the triangle vertices
    if (instance.instance_type == INSTANCE_TYPE_PATH) {
        // For now, just set default values as we're not using them
        output.tri_v0 = vec2<f32>(0.0, 0.0);
        output.tri_v1 = vec2<f32>(0.0, 0.0);
        output.tri_v2 = vec2<f32>(0.0, 0.0);
    } else {
        // For other instances, set default values
        output.tri_v0 = vec2<f32>(0.0, 0.0);
        output.tri_v1 = vec2<f32>(0.0, 0.0);
        output.tri_v2 = vec2<f32>(0.0, 0.0);
    }
    
    return output;
}

// Fragment shader - branches based on instance type
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = input.color;
    
    // Pre-compute derivatives in uniform control flow (required by WebGPU)
    // These must be calculated before any branching
    let sdf_sample = textureSample(sdf_atlas, sdf_sampler, input.uv);
    let distance = sdf_sample.r;
    let dist_grad = fwidth(distance);
    let dx = dpdx(input.local_pos.x);
    let dy = dpdy(input.local_pos.y);
    
    // Branch based on instance type
    switch (input.instance_type) {
        case INSTANCE_TYPE_RECT: {
            // Solid color rectangle - no additional processing
            return color;
        }
        case INSTANCE_TYPE_TEXT: {
            // MSDF text rendering with sharpening
            // Use pre-computed distance and gradient
            
            // Sharper rendering with tighter transition
            // Reduce the width for crisper edges
            let sharpness = 0.15;  // Smaller = sharper (was 0.2 in width/2)
            
            // Use pre-computed gradient for adaptive sharpening
            let adaptive_width = max(sharpness, dist_grad * 0.5);
            
            // Apply smoothstep with adaptive width
            let alpha = smoothstep(0.5 - adaptive_width, 0.5 + adaptive_width, distance);
            
            // Optional: Apply additional sharpening curve
            // This enhances contrast at the edges
            let sharpened_alpha = pow(alpha, 0.9);  // Slightly sharpen
            
            color.a *= sharpened_alpha;
            
            // Discard fully transparent fragments
            if (color.a < 0.01) {
                discard;
            }
            
            return color;
        }
        case INSTANCE_TYPE_PATH: {
            // Simple path rendering - just fill the entire quad
            // This is the original behavior that worked for all paths including arcs
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
                discard;
            }
            
            return color;
        }
        default: {
            // Unknown type - return magenta for debugging
            return vec4<f32>(1.0, 0.0, 1.0, 1.0);
        }
    }
    // This should never be reached due to the default case, but required for WGSL
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
});

#endif // WCN_RENDER_2D_WGSL_H
