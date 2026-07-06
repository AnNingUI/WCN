// transition.hpp — GPU-driven page transition engine for WCN UI.
//
// Architecture:
//   Page       : an Element subtree (rendered offscreen to a texture)
//   Transition : a WGSL fragment shader that blends currentTex + nextTex by progress
//   Router     : owns pages, captures textures, runs the transition render pass
//
// The engine reuses fullstack_compute's existing render path (fs_core_encode
// accepts an arbitrary target texture), so no changes to the C core are needed.
//
// Standard: C++17. Requires WebGPU (wgpu_native) headers.
#pragma once
#ifndef WCN_UI_TRANSITION_HPP
#define WCN_UI_TRANSITION_HPP

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace wcn_ui {

namespace wgpu_helper {
inline WGPUStringView sv(const char* s) {
    WGPUStringView v; v.data = s; v.length = s ? __builtin_strlen(s) : 0; return v;
}
} // namespace wgpu_helper

// ── Offscreen RGBA8 render target ────────────────────────────────────────
struct OffscreenTarget {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

    bool create(WGPUDevice device, uint32_t w, uint32_t h, WGPUTextureFormat fmt = WGPUTextureFormat_RGBA8Unorm) {
        destroy(device);
        width = w; height = h;
        WGPUTextureDescriptor td{};
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc;
        td.dimension = WGPUTextureDimension_2D;
        td.size = { w, h, 1 };
        td.format = fmt;
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        texture = wgpuDeviceCreateTexture(device, &td);
        if (!texture) return false;
        WGPUTextureViewDescriptor vd{};
        vd.format = fmt;
        vd.dimension = WGPUTextureViewDimension_2D;
        vd.aspect = WGPUTextureAspect_All;
        vd.baseMipLevel = 0; vd.mipLevelCount = 1;
        vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
        view = wgpuTextureCreateView(texture, &vd);
        return view != nullptr;
    }

    void destroy(WGPUDevice device) {
        (void)device;
        if (view) { wgpuTextureViewRelease(view); view = nullptr; }
        if (texture) { wgpuTextureRelease(texture); texture = nullptr; }
        width = height = 0;
    }
};

// ── Transition: base class ────────────────────────────────────────────────
class Transition {
public:
    virtual ~Transition() { release(); }

    const char* name() const { return name_.c_str(); }
    float duration() const { return duration_; }
    void set_duration(float seconds) { duration_ = std::max(0.05f, seconds); }

    bool ensure_pipeline(WGPUDevice device, WGPUTextureFormat output_format) {
        if (pipeline_ && output_format_ == output_format && device_ == device) return true;
        release();
        device_ = device;
        output_format_ = output_format;
        return build_pipeline(device, output_format);
    }

    void record(WGPURenderPassEncoder pass, uint32_t w, uint32_t h, float progress) {
        if (!pipeline_ || !bind_group_) return;
        struct U { float resolution[2]; float progress; float pad; } u = {
            { (float)w, (float)h }, progress, 0.0f
        };
        wgpuQueueWriteBuffer(queue_, uniform_buffer_, 0, &u, sizeof(u));
        wgpuRenderPassEncoderSetPipeline(pass, pipeline_);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group_, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    }

    void bind_textures(WGPUDevice device, WGPUQueue queue,
                       WGPUTextureView current_view, WGPUTextureView next_view,
                       WGPUSampler sampler) {
        if (!bind_group_layout_) return;
        queue_ = queue;
        WGPUBindGroupEntry entries[4]{};
        entries[0].binding = 0; entries[0].textureView = current_view;
        entries[1].binding = 1; entries[1].textureView = next_view;
        entries[2].binding = 2; entries[2].buffer = uniform_buffer_; entries[2].offset = 0; entries[2].size = 16;
        entries[3].binding = 3; entries[3].sampler = sampler;
        WGPUBindGroupDescriptor d{};
        d.layout = bind_group_layout_;
        d.entryCount = 4;
        d.entries = entries;
        if (bind_group_) wgpuBindGroupRelease(bind_group_);
        bind_group_ = wgpuDeviceCreateBindGroup(device, &d);
    }

    virtual float ease(float t) const {
        return t * t * (3.0f - 2.0f * t);
    }

protected:
    explicit Transition(std::string n, float seconds = 0.8f)
        : name_(std::move(n)), duration_(seconds) {}

    virtual const char* fragment_shader() const = 0;

    bool build_pipeline(WGPUDevice device, WGPUTextureFormat output_format) {
        WGPUBufferDescriptor ubd{};
        ubd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        ubd.size = 16;
        uniform_buffer_ = wgpuDeviceCreateBuffer(device, &ubd);
        if (!uniform_buffer_) return false;

        std::string wgsl = make_full_wgsl(fragment_shader());
        WGPUShaderSourceWGSL wsrc{};
        wsrc.chain.next = nullptr;
        wsrc.chain.sType = WGPUSType_ShaderSourceWGSL;
        wsrc.code = wgpu_helper::sv(wgsl.c_str());
        WGPUShaderModuleDescriptor smd{};
        smd.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wsrc);
        smd.label = wgpu_helper::sv("FS Transition");
        shader_module_ = wgpuDeviceCreateShaderModule(device, &smd);
        if (!shader_module_) return false;

        WGPUBindGroupLayoutEntry bgle[4]{};
        bgle[0].binding = 0; bgle[0].visibility = WGPUShaderStage_Fragment;
        bgle[0].texture.sampleType = WGPUTextureSampleType_Float;
        bgle[0].texture.viewDimension = WGPUTextureViewDimension_2D;
        bgle[1].binding = 1; bgle[1].visibility = WGPUShaderStage_Fragment;
        bgle[1].texture.sampleType = WGPUTextureSampleType_Float;
        bgle[1].texture.viewDimension = WGPUTextureViewDimension_2D;
        bgle[2].binding = 2; bgle[2].visibility = WGPUShaderStage_Fragment;
        bgle[2].buffer.type = WGPUBufferBindingType_Uniform; bgle[2].buffer.minBindingSize = 16;
        bgle[3].binding = 3; bgle[3].visibility = WGPUShaderStage_Fragment;
        bgle[3].sampler.type = WGPUSamplerBindingType_Filtering;
        WGPUBindGroupLayoutDescriptor bgld{};
        bgld.entryCount = 4; bgld.entries = bgle;
        bind_group_layout_ = wgpuDeviceCreateBindGroupLayout(device, &bgld);
        if (!bind_group_layout_) return false;

        WGPUPipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bind_group_layout_;
        pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device, &pld);
        if (!pipeline_layout_) return false;

        WGPUVertexState vs{};
        vs.module = shader_module_; vs.entryPoint = wgpu_helper::sv("vs_main");
        WGPUPrimitiveState prim{}; prim.topology = WGPUPrimitiveTopology_TriangleList;
        WGPUFragmentState fs{};
        fs.module = shader_module_; fs.entryPoint = wgpu_helper::sv("fs_main");
        fs.targetCount = 1;
        WGPUColorTargetState ct{}; ct.format = output_format;
        ct.blend = nullptr; ct.writeMask = WGPUColorWriteMask_All;
        fs.targets = &ct;
        WGPURenderPipelineDescriptor rpd{};
        rpd.layout = pipeline_layout_;
        rpd.vertex = vs;
        rpd.primitive = prim;
        rpd.fragment = &fs;
        rpd.depthStencil = nullptr;
        rpd.multisample.count = 1; rpd.multisample.mask = 0xFFFFFFFF;
        pipeline_ = wgpuDeviceCreateRenderPipeline(device, &rpd);
        return pipeline_ != nullptr;
    }

    void release() {
        if (pipeline_) { wgpuRenderPipelineRelease(pipeline_); pipeline_ = nullptr; }
        if (pipeline_layout_) { wgpuPipelineLayoutRelease(pipeline_layout_); pipeline_layout_ = nullptr; }
        if (bind_group_layout_) { wgpuBindGroupLayoutRelease(bind_group_layout_); bind_group_layout_ = nullptr; }
        if (bind_group_) { wgpuBindGroupRelease(bind_group_); bind_group_ = nullptr; }
        if (shader_module_) { wgpuShaderModuleRelease(shader_module_); shader_module_ = nullptr; }
        if (uniform_buffer_) { wgpuBufferRelease(uniform_buffer_); uniform_buffer_ = nullptr; }
    }

private:
    std::string make_full_wgsl(const char* fragment) const {
        std::string s = R"FFF(
            struct TransitionUniforms {
              resolution: vec2<f32>,
              progress: f32,
              _pad: f32,
            }
            @group(0) @binding(2) var<uniform> u: TransitionUniforms;
            @group(0) @binding(0) var currentTex: texture_2d<f32>;
            @group(0) @binding(1) var nextTex: texture_2d<f32>;
            @group(0) @binding(3) var samp: sampler;
            
            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
              var p = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -3.0),
                vec2<f32>(-3.0,  1.0),
                vec2<f32>( 3.0,  1.0)
              );
              return vec4<f32>(p[vi], 0.0, 1.0);
            };
        )FFF";
        s += fragment;
        return s;
    }

    std::string name_;
    float duration_;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    WGPUTextureFormat output_format_ = WGPUTextureFormat_Undefined;
    WGPUShaderModule shader_module_ = nullptr;
    WGPUPipelineLayout pipeline_layout_ = nullptr;
    WGPUBindGroupLayout bind_group_layout_ = nullptr;
    WGPUBindGroup bind_group_ = nullptr;
    WGPURenderPipeline pipeline_ = nullptr;
    WGPUBuffer uniform_buffer_ = nullptr;
};

// ── Built-in transitions ─────────────────────────────────────────────────

// 1. LiquidMorph: 引入双向 Domain Warping 噪声场，并对边界生成霓虹极光和色差位移
class LiquidMorphTransition : public Transition {
public:
    LiquidMorphTransition() : Transition("LiquidMorph", 1.0f) {}
protected:
    const char* fragment_shader() const override {
        return R"FFF(
            fn hash2(p: vec2<f32>) -> f32 {
              return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
            }
            fn noise(p: vec2<f32>) -> f32 {
              let i = floor(p);
              let f = fract(p);
              let u = f * f * (3.0 - 2.0 * f);
              return mix(mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
                         mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
            }
            fn fbm(p: vec2<f32>) -> f32 {
              var v = 0.0; var a = 0.5; var p_mut = p;
              for (var i = 0; i < 4; i = i + 1) {
                v = v + a * noise(p_mut);
                p_mut = p_mut * 2.0 + vec2<f32>(100.0);
                a = a * 0.5;
              }
              return v;
            }
            
            @fragment
            fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
              let uv = fragCoord.xy / u.resolution;
              let t = u.progress;
              
              // 复杂的流体域扭曲 (Domain Warping)
              let q = vec2<f32>(fbm(uv * 3.5 + vec2<f32>(0.0, t)), fbm(uv * 3.5 + vec2<f32>(5.2, 1.3 * t)));
              let r = vec2<f32>(fbm(uv * 3.5 + 4.0 * q + vec2<f32>(1.7, 9.2)), fbm(uv * 3.5 + 4.0 * q + vec2<f32>(8.3, 2.8)));
              let f = fbm(uv * 2.5 + r);
              
              let threshold = mix(1.2, -0.2, t);
              let edge_width = 0.16;
              let morph_val = smoothstep(threshold - edge_width, threshold + edge_width, f);
              
              // 边缘色差扭曲：只在过度锋面剧烈波动
              let disp = (f - 0.5) * 0.025 * (1.0 - abs(t - 0.5) * 2.0);
              let c_r = textureSample(currentTex, samp, uv + vec2<f32>(disp, 0.0)).r;
              let c_g = textureSample(currentTex, samp, uv).g;
              let c_b = textureSample(currentTex, samp, uv - vec2<f32>(disp, 0.0)).b;
              let current_col = vec4<f32>(c_r, c_g, c_b, 1.0);
              
              let n_r = textureSample(nextTex, samp, uv - vec2<f32>(disp, 0.0)).r;
              let n_g = textureSample(nextTex, samp, uv).g;
              let n_b = textureSample(nextTex, samp, uv + vec2<f32>(disp, 0.0)).b;
              let next_col = vec4<f32>(n_r, n_g, n_b, 1.0);
              
              // 青色与橙色交织的能量霓虹边缘
              let edge = 1.0 - smoothstep(0.0, edge_width * 0.4, abs(f - threshold));
              let energy_glow = vec4<f32>(0.1, 0.85, 0.95, 0.0) * edge * 2.2 * sin(t * 3.14159);
              
              return mix(current_col, next_col, morph_val) + energy_glow;
            };
        )FFF";
    }
};

// 2. RealityTear: 引入引力空间拉伸、雷暴电弧、以及深紫色的虚空裂缝能量
class RealityTearTransition : public Transition {
public:
    RealityTearTransition() : Transition("RealityTear", 0.9f) {}
protected:
    const char* fragment_shader() const override {
        return R"FFF(
            fn hash2(p: vec2<f32>) -> f32 {
              return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
            }
            fn noise(p: vec2<f32>) -> f32 {
              let i = floor(p);
              let f = fract(p); 
              let u = f * f * (3.0 - 2.0 * f);
              return mix(mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
                         mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
            }
            fn fbm(p: vec2<f32>) -> f32 {
              var v = 0.0; var a = 0.5; var p_mut = p;
              for (var i = 0; i < 3; i = i + 1) {
                v = v + a * noise(p_mut);
                p_mut = p_mut * 2.0;
                a = a * 0.5;
              }
              return v;
            }
            
            @fragment
            fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
              let uv = fragCoord.xy / u.resolution;
              let t = u.progress;
              
              // 锯齿状对角撕裂线轨迹
              let n_val = fbm(vec2<f32>(uv.y * 5.0, t * 1.5));
              let crack_center = mix(-0.25, 1.25, t) + (n_val - 0.5) * 0.14;
              let dist_to_crack = uv.x - crack_center;
              let dist_abs = abs(dist_to_crack);
              
              // 引力场拉伸形变：靠近裂缝处的空间朝轴向剧烈拉伸
              let pull = exp(-dist_abs * 14.0) * 0.045 * sin(t * 3.14159);
              let distorted_uv = uv + vec2<f32>(pull * sign(dist_to_crack), pull * (n_val - 0.5));
              
              let c = textureSample(currentTex, samp, distorted_uv);
              let n2 = textureSample(nextTex, samp, distorted_uv);
              
              let revealed = 1.0 - smoothstep(-0.015, 0.015, dist_to_crack);
              
              // 裂口虚空深紫色能量霓虹
              let glow_width = 0.02 + 0.05 * abs(sin(t * 7.0));
              let edge_glow = exp(-dist_abs / glow_width);
              let rift_color = vec4<f32>(0.75, 0.15, 1.0, 0.0) * edge_glow * 1.8;
              
              // 高频闪烁电火花线
              let spark_noise = noise(vec2<f32>(uv.y * 40.0, t * 45.0));
              let spark = smoothstep(0.93, 1.0, spark_noise) * step(dist_abs, 0.035);
              let spark_color = vec4<f32>(0.8, 0.95, 1.0, 0.0) * spark * 3.2;
              
              var col = mix(c, n2, revealed);
              col = col + rift_color + spark_color;
              return col;
            }
        )FFF";
    }
};

// 3. GlassShatter: 基于物理单元质心的 2D Voronoi 多边形破碎，包含重力模拟与折射
class GlassShatterTransition : public Transition {
public:
    GlassShatterTransition() : Transition("GlassShatter", 0.95f) {}
protected:
    const char* fragment_shader() const override {
        return R"FFF(
            fn hash22(p: vec2<f32>) -> vec2<f32> {
              let x = fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
              let y = fract(cos(dot(p, vec2<f32>(269.5, 183.3))) * 43758.5453);
              return vec2<f32>(x, y);
            }
            fn voronoi(p: vec2<f32>) -> vec3<f32> {
              let n = floor(p);
              let f = fract(p);
              var m_dist = 8.0;
              var m_cell = vec2<f32>(0.0);
              for (var j = -1; j <= 1; j = j + 1) {
                for (var i = -1; i <= 1; i = i + 1) {
                  let g = vec2<f32>(f32(i), f32(j));
                  let o = hash22(n + g);
                  let r = g + o - f;
                  let d = dot(r, r);
                  if (d < m_dist) {
                    m_dist = d;
                    m_cell = n + g;
                  }
                }
              }
              return vec3<f32>(sqrt(m_dist), m_cell.x, m_cell.y);
            }
            
            @fragment
            fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
              let uv = fragCoord.xy / u.resolution;
              let t = u.progress;
              
              let grid_scale = 7.5;
              let v = voronoi(uv * grid_scale);
              let cell_id = vec2<f32>(v.y, v.z);
              let cell_center = (cell_id + vec2<f32>(0.5)) / grid_scale;
              
              // 破碎锋面从左上到右下扫过
              let activation = (cell_center.x + cell_center.y) * 0.42;
              let shard_t = clamp((t - activation) / 0.45, 0.0, 1.0);
              
              let rand = hash22(cell_id);
              
              // 物理模拟：每个碎片由于自旋、爆炸初速度和重力向下坠落飞出
              let dir = normalize(rand - vec2<f32>(0.3, 0.1));
              let disp = dir * shard_t * shard_t * 0.35 + vec2<f32>(0.0, -shard_t * shard_t * 0.55);
              let rot = shard_t * (rand.x - 0.5) * 2.2;
              
              // 绕碎片质心旋转与平移局部坐标
              var shard_uv = uv - cell_center;
              let cos_r = cos(rot); let sin_r = sin(rot);
              shard_uv = vec2<f32>(
                shard_uv.x * cos_r - shard_uv.y * sin_r,
                shard_uv.x * sin_r + shard_uv.y * cos_r
              );
              shard_uv = shard_uv + cell_center + disp;
              
              // 玻璃内部折射效果
              let refract_uv = shard_uv + (rand - 0.5) * 0.025 * (1.0 - shard_t);
              
              let c = textureSample(currentTex, samp, refract_uv);
              let n2 = textureSample(nextTex, samp, uv);
              
              // 碎片边缘高亮闪光
              let glass_glint = exp(-abs(v.x - 0.46) * 18.0) * 0.45 * (1.0 - shard_t);
              
              // 随距离渐隐并裁切边界
              let shard_alpha = 1.0 - smoothstep(0.75, 1.0, shard_t);
              let in_bounds = step(0.0, shard_uv.x) * step(shard_uv.x, 1.0) * step(0.0, shard_uv.y) * step(shard_uv.y, 1.0);
              let final_alpha = shard_alpha * in_bounds;
              
              var col = mix(n2, c, final_alpha);
              col = col + vec4<f32>(vec3<f32>(glass_glint * 0.8), 0.0) * final_alpha;
              
              // 色散裂开
              let ca_dist = 0.016 * shard_t;
              let r_split = textureSample(currentTex, samp, refract_uv + vec2<f32>(ca_dist, 0.0)).r;
              let b_split = textureSample(currentTex, samp, refract_uv - vec2<f32>(ca_dist, 0.0)).b;
              col.r = mix(col.r, r_split, final_alpha * 0.35 * shard_t);
              col.b = mix(col.b, b_split, final_alpha * 0.35 * shard_t);
              
              return col;
            }
        )FFF";
    }
};

// 4. InkSpread: FBM 极坐标弯曲多瓣墨染，边缘色素深层堆积物理现象
class InkSpreadTransition : public Transition {
public:
    InkSpreadTransition() : Transition("InkSpread", 0.9f) {}
protected:
    const char* fragment_shader() const override {
        return R"FFF(
            fn hash2(p: vec2<f32>) -> f32 {
              return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
            }
            fn noise(p: vec2<f32>) -> f32 {
              let i = floor(p);
              let f = fract(p);
              let u = f * f * (3.0 - 2.0 * f);
              return mix(mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
                         mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
            }
            fn fbm(p: vec2<f32>) -> f32 {
              var v = 0.0; var a = 0.5; var p_mut = p;
              for (var i = 0; i < 4; i = i + 1) {
                v = v + a * noise(p_mut);
                p_mut = p_mut * 2.3;
                a = a * 0.5;
              }
              return v;
            }
            
            @fragment
            fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
              let uv = fragCoord.xy / u.resolution;
              let t = u.progress;
              
              let center = vec2<f32>(0.5, 0.5);
              let aspect = u.resolution.x / u.resolution.y;
              let corrected_uv = vec2<f32>((uv.x - center.x) * aspect, uv.y - center.y);
              let d = length(corrected_uv);
              
              // 极坐标角度扰动，生成毛细管扩散通道（纤维浸润感）
              let angle = atan2(corrected_uv.y, corrected_uv.x);
              let noise_coord = vec2<f32>(cos(angle), sin(angle)) * 2.2;
              let warp = fbm(noise_coord + vec2<f32>(t * 0.45)) * 0.24 + fbm(uv * 7.5) * 0.11;
              
              let radius = mix(-0.15, 1.35, t);
              let ink_edge = radius + warp;
              
              // 墨晕透明度映射
              let ink_val = smoothstep(ink_edge + 0.09, ink_edge - 0.09, d);
              
              let c = textureSample(currentTex, samp, uv);
              let n2 = textureSample(nextTex, samp, uv);
              
              // 模拟真实水彩/墨水扩散时，边缘干燥纤维截留色素导致的深色凝聚边缘 (Dry Edge)
              let border_width = 0.038;
              let pigment_line = (1.0 - smoothstep(0.0, border_width, abs(d - ink_edge))) * ink_val;
              
              var col = mix(c, n2, ink_val);
              
              // 融合深靛青色偏黑的墨水边界色调
              let ink_pigment_color = vec4<f32>(0.03, 0.02, 0.08, 1.0);
              col = mix(col, ink_pigment_color, pigment_line * 0.88);
              
              return col;
            }
        )FFF";
    }
};

} // namespace wcn_ui

#endif // WCN_UI_TRANSITION_HPP
