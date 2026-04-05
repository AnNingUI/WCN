// ============================================================================
// Fullstack Effects - WGSL Shaders for Physical Shadow and Filters
// ============================================================================

// ----------------------------------------------------------------------------
// Gaussian Blur Compute Shader (Separable Convolution)
// ----------------------------------------------------------------------------

struct GaussianUniforms {
    kernel_size: u32,      // 卷积核大小（奇数）
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    // kernel weights 紧随其后（动态大小，最大 63）
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: GaussianUniforms;
@group(0) @binding(3) var<storage, read> kernel_weights: array<f32>;

// 水平高斯模糊
@compute @workgroup_size(256, 1, 1)
fn gaussian_blur_h(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tex_size = textureDimensions(input_texture);
    let coord = vec2<i32>(global_id.xy);

    // 边界检查
    if (coord.x >= i32(tex_size.x) || coord.y >= i32(tex_size.y)) {
        return;
    }

    var result = vec4<f32>(0.0);
    let radius = i32(uniforms.kernel_size) / 2;

    // 应用水平卷积核
    for (var i = 0; i < i32(uniforms.kernel_size); i = i + 1) {
        let offset_x = i - radius;
        let sample_coord = clamp(
            coord + vec2<i32>(offset_x, 0),
            vec2<i32>(0),
            vec2<i32>(tex_size) - vec2<i32>(1)
        );
        let weight = kernel_weights[i];
        result = result + textureLoad(input_texture, sample_coord, 0) * weight;
    }

    textureStore(output_texture, coord, result);
}

// 垂直高斯模糊
@compute @workgroup_size(1, 256, 1)
fn gaussian_blur_v(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tex_size = textureDimensions(input_texture);
    let coord = vec2<i32>(global_id.xy);

    // 边界检查
    if (coord.x >= i32(tex_size.x) || coord.y >= i32(tex_size.y)) {
        return;
    }

    var result = vec4<f32>(0.0);
    let radius = i32(uniforms.kernel_size) / 2;

    // 应用垂直卷积核
    for (var i = 0; i < i32(uniforms.kernel_size); i = i + 1) {
        let offset_y = i - radius;
        let sample_coord = clamp(
            coord + vec2<i32>(0, offset_y),
            vec2<i32>(0),
            vec2<i32>(tex_size) - vec2<i32>(1)
        );
        let weight = kernel_weights[i];
        result = result + textureLoad(input_texture, sample_coord, 0) * weight;
    }

    textureStore(output_texture, coord, result);
}

// ----------------------------------------------------------------------------
// Filter Compute Shaders
// ----------------------------------------------------------------------------

struct FilterUniforms {
    filter_type: u32,
    param1: f32,  // 主要参数（如 brightness amount）
    param2: f32,  // 预留
    param3: f32,  // 预留
};

@group(0) @binding(0) var filter_input: texture_2d<f32>;
@group(0) @binding(1) var filter_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> filter_uniforms: FilterUniforms;

// 饱和度计算辅助函数
fn rgb_to_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

// 色相旋转辅助函数
fn hue_rotate(rgb: vec3<f32>, angle_degrees: f32) -> vec3<f32> {
    let angle_rad = radians(angle_degrees);
    let cos_a = cos(angle_rad);
    let sin_a = sin(angle_rad);

    // 旋转矩阵
    let r = vec3<f32>(
        0.299 + 0.701 * cos_a + 0.168 * sin_a,
        0.587 - 0.587 * cos_a + 0.330 * sin_a,
        0.114 - 0.114 * cos_a - 0.497 * sin_a
    );
    let g = vec3<f32>(
        0.299 - 0.299 * cos_a - 0.328 * sin_a,
        0.587 + 0.413 * cos_a + 0.035 * sin_a,
        0.114 - 0.114 * cos_a + 0.292 * sin_a
    );
    let b = vec3<f32>(
        0.299 - 0.300 * cos_a + 1.250 * sin_a,
        0.587 - 0.588 * cos_a - 1.050 * sin_a,
        0.114 + 0.886 * cos_a - 0.203 * sin_a
    );

    return vec3<f32>(dot(rgb, r), dot(rgb, g), dot(rgb, b));
}

// Sepia 转换辅助函数
fn sepia(rgb: vec3<f32>, amount: f32) -> vec3<f32> {
    let sepia_matrix = mat3x3<f32>(
        0.393, 0.769, 0.189,
        0.349, 0.686, 0.168,
        0.272, 0.534, 0.131
    );
    let sepia_color = sepia_matrix * rgb;
    return mix(rgb, sepia_color, amount);
}

// Grayscale 转换辅助函数
fn grayscale(rgb: vec3<f32>, amount: f32) -> vec3<f32> {
    let gray = vec3<f32>(rgb_to_luminance(rgb));
    return mix(rgb, gray, amount);
}

// Saturate 调整辅助函数
fn saturate(rgb: vec3<f32>, amount: f32) -> vec3<f32> {
    let luminance = rgb_to_luminance(rgb);
    let saturated = mix(vec3<f32>(luminance), rgb, amount);
    return saturated;
}

// 主 Filter compute shader
@compute @workgroup_size(8, 8, 1)
fn filter_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tex_size = textureDimensions(filter_input);
    let coord = vec2<i32>(global_id.xy);

    // 边界检查
    if (coord.x >= i32(tex_size.x) || coord.y >= i32(tex_size.y)) {
        return;
    }

    let color = textureLoad(filter_input, coord, 0);
    var result: vec4<f32>;

    switch filter_uniforms.filter_type {
        case 1u: { // brightness: 亮度调整
            let amount = filter_uniforms.param1;
            result = vec4<f32>(color.rgb * amount, color.a);
        }
        case 2u: { // contrast: 对比度调整
            let amount = filter_uniforms.param1;
            let contrast_factor = amount;
            result = vec4<f32>(
                (color.rgb - vec3<f32>(0.5)) * contrast_factor + vec3<f32>(0.5),
                color.a
            );
        }
        case 3u: { // grayscale: 灰度
            let amount = clamp(filter_uniforms.param1, 0.0, 1.0);
            result = vec4<f32>(grayscale(color.rgb, amount), color.a);
        }
        case 4u: { // hue-rotate: 色相旋转
            let angle = filter_uniforms.param1;
            result = vec4<f32>(hue_rotate(color.rgb, angle), color.a);
        }
        case 5u: { // invert: 颜色反转
            let amount = clamp(filter_uniforms.param1, 0.0, 1.0);
            result = vec4<f32>(mix(color.rgb, vec3<f32>(1.0) - color.rgb, amount), color.a);
        }
        case 6u: { // opacity: 透明度调整
            let amount = clamp(filter_uniforms.param1, 0.0, 1.0);
            result = vec4<f32>(color.rgb, color.a * amount);
        }
        case 7u: { // saturate: 饱和度调整
            let amount = filter_uniforms.param1;
            result = vec4<f32>(saturate(color.rgb, amount), color.a);
        }
        case 8u: { // sepia: 深褐色
            let amount = clamp(filter_uniforms.param1, 0.0, 1.0);
            result = vec4<f32>(sepia(color.rgb, amount), color.a);
        }
        default: { // 其他：原样输出
            result = color;
        }
    }

    // 确保颜色值在有效范围内
    result = clamp(result, vec4<f32>(0.0), vec4<f32>(1.0));

    textureStore(filter_output, coord, result);
}

// ----------------------------------------------------------------------------
// Shadow Render Shader（用于组合阴影和主渲染）
// ----------------------------------------------------------------------------

struct ShadowUniforms {
    shadow_color: vec4<f32>,
    shadow_offset: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var shadow_texture: texture_2d<f32>;
@group(0) @binding(1) var shadow_sampler: sampler;
@group(0) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;

@vertex
fn shadow_vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // 全屏 quad
    let x = f32(vertex_index % 2u) * 2.0 - 1.0;  // 0 -> -1, 1 -> 1
    let y = f32(vertex_index / 2u) * 2.0 - 1.0;  // 0 -> -1, 1 -> 1
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn shadow_fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(shadow_texture));
    let uv = frag_coord.xy / tex_size;

    // 应用阴影偏移
    let offset_uv = uv - (shadow_uniforms.shadow_offset / tex_size);

    let shadow_value = textureSample(shadow_texture, shadow_sampler, offset_uv);

    // 阴影颜色乘以阴影 alpha
    return shadow_uniforms.shadow_color * shadow_value.a;
}

// ============================================================================
// 结束 of fullstack_shader_effects.wgsl
// ============================================================================
