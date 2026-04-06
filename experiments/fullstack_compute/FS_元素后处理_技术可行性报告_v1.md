# FS 元素后处理（Post-Processing）技术可行性报告

**版本**：v1.1
**日期**：2026-04-06
**状态**：部分可行（PARTIALLY FEASIBLE）
**分析方式**：多 Agent 并行分析（技术可行性、GPU 管线设计、内存性能、API 设计）+ 顶点交互专项分析

---

## 摘要

元素后处理（After-Treatment）是 WebGPU 2D 渲染引擎 `fullstack_core` 的一项功能增强需求，允许在场景渲染完成后、通过可配置的 shader 序列对画面进行后处理（如模糊、泛光、颜色校正等）。

**分析结论**：该需求**部分可行**，无硬性技术阻塞。主要工作是工程实现而非架构探索。最关键的发现是 `fs_effects.c` 中的 ping-pong 纹理基础设施骨架已存在但尚未实现，`fs_core_encode` 只需进行结构性改造以支持多 Pass 渲染链。

---

## 目录

1. [现状分析](#1-现状分析)
2. [技术可行性](#2-技术可行性)
3. [GPU 多 Pass 管线设计](#3-gpu-多-pass-管线设计)
4. [内存与性能分析](#4-内存与性能分析)
5. [公共 API 设计](#5-公共-api-设计)
6. [分阶段实现路线图](#6-分阶段实现路线图)
7. [风险评估](#7-风险评估)
8. [附录：关键代码位置索引](#8-附录关键代码位置索引)
9. [顶点级交互设计](#9-顶点级交互设计)

---

## 1. 现状分析

### 1.1 当前渲染管线

`fs_core_encode` (`fullstack_core.c:4231-4386`) 是核心渲染入口，其 Pass 序列为：

```
fs_core_encode(encoder, target_texture, target_view)
  1. fs_flush_pending_texture_uploads()    ← 上传图集纹理
  2. fs_execute_clip_jobs()               ← compute clip mask 生成
  3. fs_update_clip_layer_uniform()       ← 更新裁剪层 Uniform
  4. wgpuQueueWriteBuffer()              ← 命令/状态/uniform 写入 GPU
  5. [compute pass]                      ← clip/transform 顶点生成（每 command 6 个顶点）
  6. [render pass]                       ← 绘制命令，带 pipeline 分批
       └── 终点：target_view（MSAA 时为 msaa_color_view + resolveTarget）
  7. fs_encode_canvas_readback_copy()    ← 可选：将 canvas 读回 CPU
  8. return
```

**关键特征**：
- 单个 `WGPURenderPass` 终点是 `target_view`
- MSAA 时使用 `msaa_color_view` + `resolveTarget`，resolve 逻辑清晰（`fullstack_core.c:4320-4322`）
- `fs_encode_canvas_readback_copy` 在 render pass 结束后立即执行（`fullstack_core.c:4380-4384`）
- WebGPU 命令编码器 (`WGPUCommandEncoder`) 可自由混合 compute pass 和 render pass

### 1.2 现有多 Pass 基础设施

#### 1.2.1 `fs_effects.c` — Ping-Pong 纹理（已声明，未实现）

`FS_EffectResources` 结构体已定义，包含：

```c
struct FS_EffectResources {
    WGPUTexture ping_pong_texture_a;   // 已声明
    WGPUTextureView ping_pong_view_a;  // 已声明
    WGPUTexture ping_pong_texture_b;   // 已声明
    WGPUTextureView ping_pong_view_b;  // 已声明
    WGPUComputePipeline gaussian_blur_h_pipeline;
    WGPUComputePipeline gaussian_blur_v_pipeline;
    WGPUBindGroupLayout gaussian_blur_bgl;
    WGPUShaderModule gaussian_shader_module;
    // ...
};
```

**现状**：`fs_effects_create_pipelines()` 是空壳函数（`fullstack_effects.c:80-91`），`fs_effects_resize()` 中 ping-pong 纹理重建逻辑被注释。这不是阻塞问题——骨架已搭好，只需填充实现。

#### 1.2.2 Compute + Render 多 Pass 协调模式（已验证）

`fs_core_encode` 已经包含 compute pass + render pass 的协调模式：

```c
WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, &compute_desc);
wgpuComputePassEncoderSetPipeline(compute_pass, core->compute_pipeline);
// ... dispatch ...
wgpuComputePassEncoderEnd(compute_pass);

// 同一 encoder 可继续 begin render pass
WGPUComputePassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_desc);
```

这套模式可直接复用于 post-processing 的多 Pass 渲染链。

### 1.3 WebGPU API 版本

代码库使用现代 WebGPU API：
- `WGPUTexelCopyBufferInfo` / `WGPUTexelCopyTextureInfo`（非旧版 `WGPUImageCopyBuffer`）
- `WGPUBufferMapCallbackInfo` + `WGPUCallbackMode` callback info 模式
- `WGPUTextureUsage_RenderAttachment` / `WGPUTextureUsage_TextureBinding` / `WGPUTextureUsage_StorageBinding`

**后处理所需操作完全支持**：
- 创建多种 usage 组合的纹理（`fullstack_core.c:2159`, `fullstack_core.c:2330`, `fullstack_core.c:2402`）
- 多次调用 `wgpuCommandEncoderBeginRenderPass` / `wgpuCommandEncoderBeginComputePass`
- BindGroup / BindGroupLayout 系统
- Compute pipeline + Render pipeline 混合使用

---

## 2. 技术可行性

### 2.1 可行性结论

**判定：部分可行（PARTIALLY FEASIBLE）**

核心渲染管线是单 pass 到 `target_texture`，但现有基础设施已提供足够的扩展能力。实现后处理需要**结构性改造**，而非不可能。

### 2.2 支持性因素

1. **Ping-pong 纹理基础设施**已在 `FS_EffectResources` 中声明
2. **Clip compute + render pass** 多 Pass 协调模式已验证可用
3. **WebGPU API 版本**完全支持所需所有操作
4. **`internal_state` 扩展点**可用于 effects 资源存储
5. **MSAA resolve 模式**与 post-processing 输入时机天然匹配——resolve 在 scene pass 内完成，post-processing 输入恒为已解析的非 MSAA 纹理
6. **`fs_effects.c` 的 getter/setter 接口**（`fs_core_get_effects_resources` / `fs_core_set_effects_resources`）已在 `fullstack_effects.h` 声明，仅需在 `fullstack_core.c` 中实现

### 2.3 阻碍因素

1. **`fs_core_encode` 需要结构性改造**：引入 `scene_texture` 中间纹理 + Pass 链重排
2. **`fs_effects.c` 的 pipeline/texture 创建是空壳**：需要完整实现
3. **Canvas readback 调用时机需要调整**：从 render pass 结束后移到 post-processing 链末端
4. **API 契约需要协调设计**：伪代码的 `begin/end` 模式与现有 `encode` 外部调用模式需要整合

### 2.4 无硬性阻塞

没有 WebGPU 功能缺失、ABI 不兼容或架构性冲突。最复杂的工作量是工程实现，而非架构探索。

### 2.5 必须改动清单

| # | 改动 | 位置 | 复杂度 |
|---|------|------|--------|
| 1 | 实现 `fs_core_get/set_effects_resources()` | `fullstack_core.c` | 小 |
| 2 | 在 `fs_core_init()` 中调用 `fs_effects_init(core)` | `fullstack_core.c` | 小 |
| 3 | 在 `fs_core_resize()` 中调用 `fs_effects_resize(core, w, h)` | `fullstack_core.c` | 小 |
| 4 | 在 `fs_core_shutdown()` 中调用 `fs_effects_destroy(core)` | `fullstack_core.c` | 小 |
| 5 | 创建 ping-pong 纹理 + view（参照 `fs_create_msaa_color_target` 模式） | `fullstack_effects.c` | 中 |
| 6 | 创建 post-processing render pipeline（全屏四边形） | `fullstack_effects.c` | 中 |
| 7 | 创建后处理 BGL + BG（纹理绑定 + uniform buffer） | `fullstack_effects.c` | 中 |
| 8 | 改造 `fs_core_encode`：scene → intermediate → post-process chain → target | `fullstack_core.c` | 中 |
| 9 | 将 `fs_encode_canvas_readback_copy` 移到 post-processing 链之后 | `fullstack_core.c` | 小 |

---

## 3. GPU 多 Pass 管线设计

### 3.1 改造后的渲染序列

```
fs_core_encode()
  ├── [COMPUTE CLIP]         ← 现有，不变
  ├── [SCENE RENDER]         ← scene → scene_texture（或 msaa_color → resolve → scene_texture）
  ├── [POST-PROCESS]          ← scene_texture → pp[A] → pp[B] → ... → swapchain
  └── [READBACK]              ← 移到 post-process 链之后
```

### 3.2 Ping-Pong 缓冲架构

复用现有 `FS_EffectResources` 的 `ping_pong_texture_a/b`：

```c
WGPUTextureUsage: WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding
Format: core->target_format（BGRA8Unorm 或 RGBA8Unorm）
Resolution: core->width × core->height
SampleCount: 1（post-processing 恒为单采样）
```

**帧索引交换**（零分支）：

```c
static inline uint32_t fs_pp_read_index(uint32_t frame_index) {
    return frame_index & 1u;        // 0 = A, 1 = B
}
static inline uint32_t fs_pp_write_index(uint32_t frame_index) {
    return (frame_index + 1u) & 1u; // 与读索引相反
}
```

**三缓冲变体**（可选，用于需要多写 pass 的场景，如 bloom extract + blur 同帧）：

```c
WGPUTexture ping_pong_texture_c;   // scratch buffer
WGPUTextureView ping_pong_view_c;
```

### 3.3 Hook 位置

在 `fullstack_core.c:4379`，`wgpuRenderPassEncoderEnd(pass)` 之后：

```c
// 当前行为：render pass 结束后直接 return
// 改造后：

// 1. 确定 post-processing 输入源（MSAA resolved 或 direct）
WGPUTexture scene_texture = (core->render_sample_count > 1u)
    ? core->msaa_color_texture
    : target_texture;
WGPUTextureView scene_view = (core->render_sample_count > 1u)
    ? core->msaa_color_view
    : target_view;

// 2. 调用 post-processing 链
if (!fs_postprocess_encode(core, encoder,
        scene_texture, scene_view,
        swapchain_texture, swapchain_view,
        core->frame_index)) {
    fs_mark_context_lost(core);
    return false;
}

// 3. Canvas readback 移到 post-process 链之后
fs_encode_canvas_readback_copy(core, encoder, target_texture);
```

### 3.4 Pass 执行循环

```c
bool fs_postprocess_encode(
    FS_Core* core, WGPUCommandEncoder encoder,
    WGPUTexture scene_texture, WGPUTextureView scene_view,
    WGPUTexture swapchain_texture, WGPUTextureView swapchain_view,
    uint32_t frame_index)
{
    FS_EffectResources* fx = fs_core_get_effects_resources(core);

    // 零优化：无 effect 时直接 blit
    if (!fx || !fs_effects_has_active_effects(fx)) {
        fs_blit_texture(core, encoder, scene_view, swapchain_view);
        return true;
    }

    uint32_t pass_count = fx->pass_count;

    // Pass 0: scene → pp[write]
    uint32_t write_idx = fs_pp_write_index(frame_index);
    WGPUTextureView pp_write_view = (write_idx == 0) ? fx->ping_pong_view_a : fx->ping_pong_view_b;

    if (!fs_encode_postprocess_pass(core, encoder,
            scene_view, pp_write_view, &fx->passes[0])) {
        return false;
    }

    // 中间 Pass：pp[read] → pp[write]，交替交换
    uint32_t read_idx = write_idx;
    for (uint32_t p = 1; p < pass_count - 1; p++) {
        write_idx = fs_pp_write_index(frame_index + p);
        WGPUTextureView pp_read_view  = (read_idx == 0) ? fx->ping_pong_view_a : fx->ping_pong_view_b;
        pp_write_view = (write_idx == 0) ? fx->ping_pong_view_a : fx->ping_pong_view_b;

        if (!fs_encode_postprocess_pass(core, encoder,
                pp_read_view, pp_write_view, &fx->passes[p])) {
            return false;
        }
        read_idx = write_idx;
    }

    // 最终 Pass：pp[read] → swapchain（直接写屏）
    WGPUTextureView pp_read_view_final = (read_idx == 0) ? fx->ping_pong_view_a : fx->ping_pong_view_b;
    if (!fs_encode_postprocess_pass_to_target(core, encoder,
            pp_read_view_final, swapchain_view, swapchain_texture,
            &fx->passes[pass_count - 1])) {
        return false;
    }

    return true;
}
```

### 3.5 Shader 接口

#### Bind Group Layout

标准后处理 BGL（3 个 binding）：

```c
WGPUBindGroupLayoutEntry pp_bgl[] = {
    // Binding 0: Uniform buffer
    {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = { .type = WGPUBufferBindingType_Uniform, .hasDynamicOffset = false }
    },
    // Binding 1: 输入纹理（只读采样）
    {
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = {
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
            .multisampled = false
        }
    },
    // Binding 2: 采样器
    {
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = { .type = WGPUSamplerBindingType_NonFiltering }
    }
};
```

扩展 BGL（4 binding，用于多输入 pass，如 bloom composite）增加 Binding 3 的纹理 binding。

#### 全屏四边形 Pipeline

无顶点缓冲，VertexID 生成四边形：

```c
WGPURenderPipelineDescriptor desc = {
    .layout = pipeline_layout,
    .vertex = {
        .module = vs_module,
        .entryPoint = WGPU_STRING_VIEW("vs_main"),
        .bufferCount = 0,    // 无顶点缓冲
        .buffers = NULL
    },
    .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleStrip,
        .stripIndexFormat = WGPUIndexFormat_Undefined,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_None
    },
    .depthStencil = NULL,
    .multisample = { .count = 1, .mask = 0xFFFFFFFF, .alphaToCoverageEnabled = false },
    .fragment = &fs_state
};
```

对应的 WGSL Vertex Shader：

```wgsl
struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let u = f32(vi & 1u);
    let v = f32((vi >> 1u) & 1u);
    out.clip_pos = vec4f(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0, 1.0);
    out.uv = vec2f(u, v);
    return out;
}
```

#### Per-Pass Render Pass 编码

```c
static bool fs_encode_postprocess_pass(
    FS_Core* core, WGPUCommandEncoder encoder,
    WGPUTextureView src_view, WGPUTextureView dst_view,
    WGPUTexture dst_texture, FS_PostprocessPass* pass)
{
    WGPURenderPassColorAttachment color = {
        .view = dst_view,
        .resolveTarget = NULL,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = { .r = 0, .g = 0, .b = 0, .a = 1 }
    };
    WGPURenderPassDescriptor pass_desc = {
        .label = pass->debug_name,
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
        .depthStencilAttachment = NULL
    };

    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
    wgpuRenderPassEncoderSetPipeline(rp, pass->pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, pass->bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(rp, 4, 1, 0, 0);   // 全屏四边形
    wgpuRenderPassEncoderEnd(rp);
    return true;
}
```

### 3.6 Uniform 设计

**固定 Header（48 字节）**：

```c
typedef struct FS_PostprocessUniforms {
    float viewport[2];      // 纹理分辨率
    float texel_size[2];   // 1.0 / viewport
    float time;            // 帧时间（秒，动画用）
    float padding[3];
} FS_PostprocessUniforms;
_Static_assert(sizeof(FS_PostprocessUniforms) == 48u, "");
```

**Pass 特定扩展**（追加到同一 buffer 的固定偏移处）：

```c
typedef struct FS_GaussianBlurUniforms {
    float direction[2];     // {1,0} 水平 H，{0,1} 垂直 V
    float sigma;
    float kernel_size;
    float weights[64];     // FS_GAUSSIAN_KERNEL_MAX_SIZE
} FS_GaussianBlurUniforms;
```

### 3.7 数据结构

**Pass 定义**：

```c
typedef struct FS_PostprocessPass {
    WGPURenderPipeline pipeline;
    WGPUBindGroupLayout pipeline_bgl;
    WGPUBindGroup bind_group;
    WGPUBuffer uniform_buffer;
    size_t uniform_size;
    const char* debug_name;
    WGPUTextureView input_view0;
    WGPUTextureView input_view1;   // NULL if not used
    WGPUSampler sampler;
} FS_PostprocessPass;
```

**Effect Resources 扩展**：

```c
struct FS_EffectResources {
    // === Ping-pong 缓冲 ===
    WGPUTexture ping_pong_texture_a;
    WGPUTextureView ping_pong_view_a;
    WGPUTexture ping_pong_texture_b;
    WGPUTextureView ping_pong_view_b;
    WGPUTexture ping_pong_texture_c;    // 可选 scratch buffer
    WGPUTextureView ping_pong_view_c;

    // === Post-processing 状态 ===
    bool postprocess_enabled;
    FS_PostprocessPass* passes;
    uint32_t pass_count;
    uint32_t pass_capacity;
    uint32_t frame_index;
    float postprocess_time_seconds;
    uint32_t width;
    uint32_t height;

    // === 共享资源 ===
    WGPUSampler postprocess_sampler;
    WGPUBuffer shared_uniform_buffer;
    WGPUBindGroupLayout shared_bgl;       // 标准 3-binding
    WGPUBindGroupLayout shared_bgl_ext;  // 扩展 4-binding

    // === 预构建管线 ===
    WGPUShaderModule fullscreen_vs_module;
    WGPURenderPipeline blit_pipeline;    // identity blit（无 effect 时快速路径）
};
```

---

## 4. 内存与性能分析

### 4.1 当前显存占用（无后处理，1920×1080）

| 资源 | 格式 | 大小 |
|------|------|------|
| Canvas shadow（CPU） | RGBA8 | 8.3 MB |
| Clip mask texture | R8Unorm | 2.1 MB |
| MSAA color（4x） | RGBA8 | 33.1 MB |
| Image atlas | RGBA8 | 16 MB |
| Glyph atlas | RGBA8 | 16 MB |
| Canvas readback buffer | RGBA8 | 8.3 MB |

### 4.2 后处理增量（Ping-Pong 缓冲）

| 资源 | 格式 | 大小 |
|------|------|------|
| ping_pong_texture_a | RGBA8Unorm | 8.3 MB |
| ping_pong_texture_b | RGBA8Unorm | 8.3 MB |
| **合计增量** | | **+16.6 MB** |

4K（3840×2160）下约 33 MB/buffer，完全可接受。

### 4.3 分配策略

遵循现有 `fs_ensure_canvas_shadow` 惰性分配模式：

```c
static bool fs_effects_ensure_ping_pong_capacity(FS_EffectResources* res,
    uint32_t width, uint32_t height, WGPUDevice device)
{
    // 同尺寸早期退出，无重新分配
    if (res->width == width && res->height == height &&
        res->ping_pong_texture_a && res->ping_pong_texture_b) {
        return true;
    }
    fs_effects_destroy_ping_pong_textures(res);
    res->width = width;
    res->height = height;
    return fs_effects_create_ping_pong_textures(res, device);
}
```

纹理创建规格：
- `format = core->target_format`（避免格式转换）
- `usage = RenderAttachment | TextureBinding`
- `mipLevelCount = 1`（后处理无需 mip）
- `sampleCount = 1`（恒为单采样）

### 4.4 带宽分析（1920×1080）

| 操作 | 读 | 写 | 合计 |
|------|-----|-----|------|
| 基础场景渲染 | — | 8.3 MB | 8.3 MB |
| MSAA resolve（4x） | 33.1 MB | 8.3 MB | 41.4 MB |
| Post-process Pass 1 | 8.3 MB | 8.3 MB | 16.6 MB |
| Post-process Pass 2 | 8.3 MB | 8.3 MB | 16.6 MB/pair |
| 最终复制到屏幕 | 8.3 MB | — | 8.3 MB |

**每像素公式**：`N` 个后处理 Pass = `(2N) × width × height × 4` 字节

按分辨率分项：

| 分辨率 | 1 对 Pass（H+V） | 2 对 Pass | 3 对 Pass |
|--------|-----------------|-----------|-----------|
| 640×360 | 3.6 MB | 7.2 MB | 10.8 MB |
| 1280×720 | 14.0 MB | 28.0 MB | 42.0 MB |
| **1920×1080** | **33.2 MB** | **66.4 MB** | **99.6 MB** |
| 2560×1440 | 58.8 MB | 117.6 MB | 176.4 MB |
| 3840×2160 | 132.8 MB | 265.6 MB | 398.4 MB |

**结论**：即使 4K 下，一对 H+V blur Pass 约 266 MB 总带宽，在现代独立 GPU（200-500 GB/s 带宽）上耗时 **< 1ms**。后处理完全是 GPU 端操作，无 CPU/GPU 传输开销。

### 4.5 MSAA 交互

**处理策略**：Post-processing 操作于 **MSAA resolve 后的非 MSAA 纹理**。

```
Render scene → MSAA texture → [auto-resolve] → Canvas texture
Post-process Canvas texture → Ping-pong → ... → Screen
```

- MSAA 纹理（`msaa_color_texture`）与 Ping-pong 缓冲**完全独立**
- MSAA 采样数**不影响** ping-pong 缓冲大小
- MSAA 4x 额外显存 33.1 MB，resolve 额外带宽 33.1 MB

### 4.6 零优化

无 effect 配置时，bypass 整个后处理链：

```c
bool fs_effects_has_active_effects(const FS_EffectResources* res) {
    return res->shadow_enabled;   // 或 equivalent active effect count
}

// 在 fs_postprocess_encode 中：
if (!fs_effects_has_active_effects(fx)) {
    fs_blit_texture(core, encoder, scene_view, swapchain_view);
    return true;   // 零额外纹理操作，零额外 Draw Call
}
```

零优化时行为与当前完全一致，无任何额外开销。

### 4.7 Resize 集成

在 `fs_core_resize` 中，`fs_effects_resize` 调用位置为 `fs_create_msaa_color_target` 之后：

```c
// fs_core_resize 中的调用顺序：
// ... canvas shadow, readback buffer, clip mask, msaa color target ...
fs_create_msaa_color_target(core);
// ← 在此处插入：
FS_EffectResources* effects = fs_core_get_effects_resources(core);
if (effects) {
    if (!fs_effects_resize(core, width, height)) {
        fs_mark_context_lost(core);
        return;
    }
    // 纹理尺寸变化后重建 bind group
    fs_effects_rebuild_bind_groups(effects, core->device);
}
```

### 4.8 线程安全

无额外同步需求：
- 所有操作在渲染线程上执行
- WebGPU 命令编码通过单一 `WGPUCommandEncoder` 序列化
- Ping-pong 纹理不跨帧共享，不跨线程共享
- `fs_effects_init/destroy/resize` 均在 `fs_core_init/shutdown/resize` 中调用，单线程

---

## 5. 公共 API 设计

API 头文件：`src/Draft/postprocess_api_design.h`

设计原则：与现有 `fs_*` 命名风格完全一致，uniform buffer 支持自定义扩展，生命周期清晰。

### 5.1 数据结构

```c
// Uniform Header（WGSL 对应 struct PostprocessUniforms）
typedef struct FS_PostprocessUniformHeader {
    float resolution[2];         // 输出纹理宽高（像素）
    float time_delta;            // 距离上一帧的秒数
    uint32_t frame_count;        // 帧计数，每帧递增
    uint32_t channel_count;      // 1 = 仅源纹理；2 = 同时读上一 Pass
    uint32_t _pad0, _pad1, _pad2;
    float channel_resolution[4][2];  // 每个通道的纹理尺寸
    // 自定义 f32 值追加在此处
} FS_PostprocessUniformHeader;

// Shader 类型
typedef enum FS_PostprocessShaderType {
    FS_POSTPROCESS_SHADER_TYPE_FULLSCREEN_QUAD = 0,
} FS_PostprocessShaderType;

// 渲染模式
typedef enum FS_PostprocessRenderMode {
    FS_POSTPROCESS_RENDER_MODE_REPLACE = 0,  // 直接覆盖写入
    FS_POSTPROCESS_RENDER_MODE_BLEND  = 1,  // alpha 混合
} FS_PostprocessRenderMode;
```

### 5.2 Shader 生命周期

```c
// 从 WGSL 字符串编译
FS_PostprocessShader* fs_postprocess_shader_create_from_wgsl(
    WGPUDevice device,
    const char* wgsl_code,    // 包含 vs_main() 和 fs_main() 的 WGSL 代码
    const char* label);

// 从 .wgsl 文件编译
FS_PostprocessShader* fs_postprocess_shader_create_from_file(
    WGPUDevice device,
    const char* path,
    const char* label);

// 释放
void fs_postprocess_shader_destroy(FS_PostprocessShader* shader);
```

### 5.3 Uniform Buffer 管理

```c
// 创建（可选，不创建则使用默认零初始化 Header）
FS_PostprocessUniformBuffer* fs_postprocess_uniform_buffer_create(
    WGPUDevice device,
    uint32_t custom_f32_count);   // Header 后的自定义 f32 数量

void fs_postprocess_uniform_buffer_destroy(FS_PostprocessUniformBuffer*);

// 写入自定义值
bool fs_postprocess_uniform_buffer_set_f32(
    FS_PostprocessUniformBuffer*, uint32_t index, float value);

bool fs_postprocess_uniform_buffer_set_vec2(
    FS_PostprocessUniformBuffer*, uint32_t index, float x, float y);

bool fs_postprocess_uniform_buffer_set_vec4(
    FS_PostprocessUniformBuffer*, uint32_t index,
    float x, float y, float z, float w);
```

### 5.4 帧生命周期

```c
// 开始一帧后处理（在场景渲染完成后调用）
bool fs_postprocess_begin(
    FS_Core* core,
    uint32_t width, uint32_t height,
    WGPUTextureView source_view);  // 场景输出纹理视图，NULL = 跳过

// 添加一个 Pass（可多次调用）
bool fs_postprocess_add_shader(
    FS_Core* core,
    FS_PostprocessShader* shader,
    FS_PostprocessUniformBuffer* uniforms,  // NULL = 默认零值
    FS_PostprocessRenderMode mode);

// 结束帧（编码所有 Pass 到 CommandBuffer）
bool fs_postprocess_end(
    FS_Core* core,
    WGPUCommandEncoder encoder,
    WGPUTexture target_texture,
    WGPUTextureView target_view);

// 调整内部纹理尺寸（canvas resize 时调用）
bool fs_postprocess_resize(FS_Core* core, uint32_t width, uint32_t height);

// 初始化/销毁（通常由 fs_core_init/shutdown 自动调用）
bool fs_postprocess_init(FS_Core* core);
void fs_postprocess_destroy(FS_Core* core);
```

### 5.5 用户使用示例

```c
// === 用户代码 ===
fs_postprocess_begin(core, width, height, scene_view);

// 第一个 Pass：高斯模糊水平
fs_postprocess_uniform_buffer_set_vec4(blur_h_ub, 0, 1.0f, 0.0f, 0.0f, 0.0f); // direction = {1,0}
fs_postprocess_add_shader(core, blur_h_shader, blur_h_ub,
    FS_POSTPROCESS_RENDER_MODE_REPLACE);

// 第二个 Pass：高斯模糊垂直
fs_postprocess_uniform_buffer_set_vec4(blur_v_ub, 0, 0.0f, 1.0f, 0.0f, 0.0f); // direction = {0,1}
fs_postprocess_add_shader(core, blur_v_shader, blur_v_ub,
    FS_POSTPROCESS_RENDER_MODE_REPLACE);

// 第三个 Pass：泛光合成
fs_postprocess_uniform_buffer_set_f32(bloom_ub, 0, 1.2f); // intensity
fs_postprocess_add_shader(core, bloom_composite_shader, bloom_ub,
    FS_POSTPROCESS_RENDER_MODE_BLEND);

// ... 渲染正常 UI 命令 ...
fs_postprocess_end(core, encoder, swapchain_texture, swapchain_view);
```

### 5.6 Shader WGSL 约定

用户提供的 WGSL shader 必须遵循以下接口约定：

```wgsl
// === Vertex Shader ===
struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput;

// === Fragment Shader ===
@group(0) @binding(0) var s_sampler: sampler;
@group(0) @binding(1) var t_source: texture_2d<f32>;
// binding 2 = 可选第二输入纹理（bloom composite 等）
// binding 3 = 可选第三输入纹理

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(t_source, s_sampler, in.uv);
    // ... 后处理逻辑 ...
    return vec4f(r, g, b, a);
}
```

---

## 6. 分阶段实现路线图

### Phase 0：基础设施填充（2~3 天）

**目标**：将 `fs_effects.c` 的空壳填充为可工作的代码。

- [ ] 实现 `fs_core_get/set_effects_resources()`（`fullstack_core.c`）
- [ ] 在 `fs_core_init()` 中调用 `fs_effects_init(core)`
- [ ] 在 `fs_core_shutdown()` 中调用 `fs_effects_destroy(core)`
- [ ] 在 `fs_core_resize()` 中调用 `fs_effects_resize(core, w, h)`
- [ ] 实现 `fs_effects_create_ping_pong_textures()`（参照 `fs_create_msaa_color_target`）
- [ ] 实现 `fs_effects_destroy_ping_pong_textures()`
- [ ] 创建共享 sampler（clamp-to-edge, non-filtering）

### Phase 1：间接渲染验证（1~2 天）

**目标**：验证 scene_texture 中间渲染路径正确，不引入任何 effect。

- [ ] 场景渲染目标从直接写入 `target_view` 改为写入中间 `scene_texture`
- [ ] 添加 blit-only fast path（无 effect 时：`scene_texture` → `swapchain`，直接复制）
- [ ] 验证 MSAA resolve 正确工作
- [ ] 验证 resize 时纹理重建正确
- [ ] 端到端渲染测试通过

### Phase 2：多 Pass 管线（2~3 天）

**目标**：实现完整的多 Pass ping-pong 渲染链。

- [ ] 实现 `fs_postprocess_encode()` 主循环
- [ ] 创建全屏四边形 pipeline（VertexID，无 VB）
- [ ] 实现 `fs_encode_postprocess_pass()` 单 Pass 编码函数
- [ ] 实现 `fs_encode_postprocess_pass_to_target()`（最终 Pass 写屏）
- [ ] 每个 Pass 支持独立的 uniform buffer 写入
- [ ] 零 Pass 快速路径（shader list 为空时 bypass）
- [ ] 实现 `fs_pp_upload_uniforms()` —— 将 VertexDeformUniforms 上传到 GPU

### Phase 3：API + Shader 编译（1~2 天）

**目标**：完成公共 API 并验证首个真实 effect。

- [ ] 实现 `fs_postprocess_shader_create_from_wgsl()`
- [ ] 实现 `fs_postprocess_shader_create_from_file()`
- [ ] 实现 `fs_postprocess_uniform_buffer_*` 系列函数
- [ ] 实现 `fs_postprocess_begin/add/end()` 帧生命周期
- [ ] **实现顶点交互 API**：`fs_pp_register_effect_type()`, `fs_pp_effect_enable/disable()`, `fs_pp_effect_set_f32/vec2/vec4()`, `fs_pp_set_mouse()`
- [ ] 实现 `fullstack_postprocess_vertex.wgsl` 内置函数库 + 路由
- [ ] 第一个真实 effect：blur H/V（或 bloom）
- [ ] 端到端集成测试

### Phase 4：扫尾（1 天）

**目标**：生产就绪。

- [ ] 将 `fs_encode_canvas_readback_copy` 移到 post-processing 链之后
- [ ] resize 时 bind group 重建逻辑
- [ ] WebGPU validation layer 全量通过
- [ ] 文档和示例代码
- [ ] 性能基准测试

---

## 7. 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|----------|
| `fs_core_encode` 结构性重构 | 中 | Phase 1 先做 scene_texture → target blit，验证后逐步加 effect |
| MSAA resolve 与 post-process 顺序 | 中 | resolve 在 scene pass 内完成，post-process 输入恒为 resolved |
| 显存翻倍 | 中 | 4K 下约 33 MB/buffer，完全可接受。零 Pass 时跳过 ping-pong 创建 |
| API 契约变化 | 中 | 所有调用方需更新，但影响范围可控（仅 render loop） |
| 空壳实现依赖 | 低 | 骨架已搭好，Phase 0 填充实现即可 |
| 读回时机 | 低 | canvas readback 移到 post-process 链末端即可 |
| 线程安全 | 低 | 所有操作在单线程上，通过 WebGPU queue 序列化 |
| 顶点交互 shader 路由表维护 | 低 | switch-case 路由在 WGSL 端，新增 effect 需同步改路由（但频率极低） |

---

## 8. 附录：关键代码位置索引

| 主题 | 文件 | 行号 |
|------|------|------|
| `fs_core_encode` 主函数 | `fullstack_core.c` | 4231 |
| Scene render pass（MSAA + resolve） | `fullstack_core.c` | 4320-4330 |
| Scene render pass 结束 | `fullstack_core.c` | 4379 |
| Canvas readback | `fullstack_core.c` | 4380-4384 |
| `fs_core_resize` | `fullstack_core.c` | ~3740 |
| `fs_effects_create_pipelines`（空壳） | `fullstack_effects.c` | 80-91 |
| `FS_EffectResources` 声明 | `fullstack_effects.c` | 12-29 |
| `fs_effects_resize` | `fullstack_effects.c` | ~146 |
| `FS_Core` 结构体 | `fullstack_core_private.h` | 415-562 |
| `FS_Command` 布局 | `fullstack_core_gpu_layout.h` | 54-69 |
| `FS_VertexGPU` 布局 | `fullstack_core_gpu_layout.h` | 94-106 |
| MSAA 纹理创建 | `fullstack_core.c` | ~2402 |
| WebGPU API 版本特征 | 整个代码库 | — |
| `fs_effects_get/set` 声明 | `fullstack_effects.h` | 69-70 |
| Ping-pong 纹理基础设施 | `FS_EffectResources` | `fullstack_effects.c:13-16` |

### 产出文件索引

| 文件 | 内容 |
|------|------|
| `src/Draft/fs_元素后处理.伪代码` | 原始需求伪代码 |
| `src/Draft/postprocess_analysis_tech.md` | 技术可行性详细分析 |
| `src/Draft/postprocess_pipeline_design.md` | GPU 管线设计详案 |
| `src/Draft/postprocess_memory_perf.md` | 内存与性能详细分析 |
| `src/Draft/postprocess_api_design.h` | 公共 C API 头文件 |
| `src/Draft/FS_元素后处理_技术可行性报告_v1.md` | 本文档（综合报告 v1.1，含顶点交互设计） |

---

## 9. 顶点级交互设计

> 本节回答一个原报告中未覆盖的核心问题：后处理阶段如何响应鼠标/触摸等交互输入，实现顶点级的画面变形（涟漪、凸镜、波动、色差等）。

### 9.1 设计目标

1. **持久可扩展**：不只为某一特定效果设计，支持任意数量的顶点变形 effect 叠加
2. **数据驱动**：通过 uniform buffer 传递所有参数，C 端 API 驱动，无需修改 shader 即可添加新 effect
3. **UI 交互完整闭环**：鼠标/触摸输入 → C 端事件 → uniform 更新 → vertex shader 变形 → 画面响应
4. **零额外开销**：无 effect 激活时，uniform buffer 内容忽略，vertex shader 走快速路径

### 9.2 统一 Uniform 布局

```c
/**
 * 后处理顶点变形通用 Uniform。
 *
 * 设计原则：
 * - 每个 effect 占固定大小的 parameter slot
 * - slots[] 数组注册 effect，支持任意数量叠加
 * - vertex shader 内置 effect 函数库，通过 slots[].type 路由
 * - 用户可在 C 端用 API 注册新 effect type（函数指针存元数据）
 *
 * Vertex Shader 中对应的 effect 函数签名：
 *   vec2f effect_XXX(vec2f pos, vec2f uv, const f32 params[32])
 *   返回变形后的 clip-space 位置
 */
#define FS_PP_MAX_EFFECTS      8u
#define FS_PP_MAX_PARAM_F32   32u   // 每个 effect 最多 32 个 float 参数

typedef struct FS_PostprocessVertexDeformUniforms {
    // === 全局控制 ===
    float viewport[2];       // 纹理分辨率（像素）
    float time;              // 帧时间（秒，动画用）
    float mouse[2];          // 当前鼠标位置（像素，左上原点）
    float mouse_click[2];    // 最近一次按下位置
    float mouse_down;        // 1.0 = 按下, 0.0 = 未按下
    float padding0;

    // === 活跃 Effect 列表 ===
    uint32_t effect_count;   // 实际使用的 effect 数量
    uint32_t effect_stride;  // sizeof(FS_PPEffectParam)，用于 shader 偏移

    // === Effect 参数表（紧排，最多 8 个 slot） ===
    FS_PPEffectParam slots[FS_PP_MAX_EFFECTS];
} FS_PostprocessVertexDeformUniforms;

_Static_assert(sizeof(FS_PostprocessVertexDeformUniforms) == 240u, "");

/**
 * 单个 effect 的参数块（固定 128 字节）。
 * type 决定 vertex shader 中使用哪套计算逻辑。
 * params[0..31] 是 effect 私有的 float 参数，含义由 type 决定。
 * weight 控制多 effect 叠加时的混合权重（0~1）。
 */
typedef struct FS_PPEffectParam {
    uint32_t type;          // FS_PPEffectType_* 枚举值
    uint32_t enabled;       // 1 = 激活, 0 = 跳过
    float weight;            // 叠加权重 [0, 1]
    float padding0;
    float params[FS_PP_MAX_PARAM_F32];  // effect 私有参数
} FS_PPEffectParam;

/**
 * 内置 effect type 枚举。
 * CUSTOM_0 及以后供用户扩展。
 */
typedef enum FS_PPEffectType {
    FS_PP_EFFECT_NONE           = 0,
    FS_PP_EFFECT_RIPPLE         = 1,   // 水波涟漪
    FS_PP_EFFECT_BULGE          = 2,   // 凸镜/凹镜
    FS_PP_EFFECT_WAVE           = 3,   // 水平/垂直波动
    FS_PP_EFFECT_SWIRL          = 4,   // 漩涡
    FS_PP_EFFECT_CHROMATIC_AB   = 5,   // 色差（顶点级 RGB 分离）
    FS_PP_EFFECT_PERSPECTIVE    = 6,   // 透视四角变换
    FS_PP_EFFECT_CUSTOM_0       = 64,  // 用户可从 CUSTOM_0 开始注册新 type
} FS_PPEffectType;
```

### 9.3 Vertex Shader 函数库

`fullstack_postprocess_vertex.wgsl` —— 内置顶点变形函数库 + 统一路由入口：

```wgsl
// ============================================================
// fullstack_postprocess_vertex.wgsl
// 内置顶点变形函数库 + 统一路由入口
// ============================================================

// --- 工具函数 ---
fn pp_lerp(a: vec2f, b: vec2f, t: f32) -> vec2f { return a * (1.0-t) + b * t; }

fn pp_dist_to_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a; let ba = b - a;
    let h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa - ba*h);
}

// ============================================================
// Effect 函数（每个 effect 一个函数，固定签名）
// 参数 params[] 布局见 9.2 节各 effect 的 params 说明
// ============================================================

fn effect_ripple(pos: vec2f, uv: vec2f, p: const f32) -> vec2f {
    // p[0]=center_x(UV), p[1]=center_y(UV), p[2]=amplitude, p[3]=frequency, p[4]=speed, p[5]=decay
    let c = vec2f(p[0], p[1]);
    let d = length(uv - c);
    let wave = sin(d * p[3] * 6.2832 - uniforms.time * p[4]) * p[2] * exp(-d * p[5]);
    let dir = normalize(uv - c + vec2f(0.0001));
    return pos + dir * wave;
}

fn effect_bulge(pos: vec2f, uv: vec2f, p: const f32) -> vec2f {
    // p[0]=center_x(UV), p[1]=center_y(UV), p[2]=radius, p[3]=strength(-1=凹,+1=凸)
    let c = vec2f(p[0], p[1]);
    let d = length(uv - c);
    let t = clamp(1.0 - d / p[2], 0.0, 1.0);
    let bulge = t * t * p[3] * 0.3;
    let dir = normalize(uv - c + vec2f(0.0001));
    return pos + dir * bulge;
}

fn effect_wave(pos: vec2f, uv: vec2f, p: const f32) -> vec2f {
    // p[0]=axis(0=X,1=Y), p[1]=amplitude, p[2]=frequency, p[3]=speed
    let axis = u32(p[0]);
    let coord = (axis == 0u) ? uv.x : uv.y;
    let wave = sin(coord * p[2] * 6.2832 + uniforms.time * p[3]) * p[1];
    if (axis == 0u) { pos.x += wave; } else { pos.y += wave; }
    return pos;
}

fn effect_swirl(pos: vec2f, uv: vec2f, p: const f32) -> vec2f {
    // p[0]=center_x(UV), p[1]=center_y(UV), p[2]=radius, p[3]=angle, p[4]=decay
    let c = vec2f(p[0], p[1]);
    let d = length(uv - c);
    let t = clamp(1.0 - d / p[2], 0.0, 1.0);
    let theta = p[3] * t * exp(-d * p[4]);
    let s = sin(theta); let co = cos(theta);
    let centered = uv - c;
    let rotated = vec2f(centered.x*co - centered.y*s, centered.x*s + centered.y*co);
    return pos + (rotated - centered) * 2.0;
}

fn effect_perspective(pos: vec2f, uv: vec2f, p: const f32) -> vec2f {
    // p[0..3]: top-left, top-right, bottom-left, bottom-right 各 (x,y) 偏移
    let tl = vec2f(p[0], p[1]);
    let tr = vec2f(p[2], p[3]);
    let bl = vec2f(p[4], p[5]);
    let br = vec2f(p[6], p[7]);
    let top = pp_lerp(tl, tr, uv.x);
    let bot = pp_lerp(bl, br, uv.x);
    let offset = pp_lerp(top, bot, uv.y);
    return pos + offset;
}

// ============================================================
// 统一路由入口
// ============================================================
fn apply_vertex_deform(pos: vec2f, uv: vec2f) -> vec2f {
    var result = pos;
    for (var i = 0u; i < effect_count; i++) {
        if (slots[i].enabled == 0u) { continue; }
        let p = slots[i].params;
        var deformed = pos;
        switch (slots[i].type) {
            case 1u:  deformed = effect_ripple(pos, uv, p);    break;
            case 2u:  deformed = effect_bulge(pos, uv, p);     break;
            case 3u:  deformed = effect_wave(pos, uv, p);       break;
            case 4u:  deformed = effect_swirl(pos, uv, p);     break;
            case 6u:  deformed = effect_perspective(pos, uv, p); break;
            default: break;
        }
        result = pp_lerp(result, deformed, slots[i].weight);
    }
    return result;
}

// ============================================================
// 主入口：全屏四边形顶点生成 + 变形
// ============================================================
struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) ab_r_offset: vec2f,   // 色差：R 通道 UV 偏移
    @location(2) ab_b_offset: vec2f,   // 色差：B 通道 UV 偏移
};

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let u = f32(vi & 1u);
    let v = f32((vi >> 1u) & 1u);

    var pos = vec2f(u * 2.0 - 1.0, 1.0 - v * 2.0);
    let uv = vec2f(u, v);

    // 应用顶点变形
    pos = apply_vertex_deform(pos, uv);

    // 色差 effect：写入 UV 偏移到 varyings，FS 分别采样不同 UV
    var r_off = vec2f(0.0);
    var b_off = vec2f(0.0);
    for (var i = 0u; i < effect_count; i++) {
        if (slots[i].type == 5u && slots[i].enabled == 1u) {
            // p[0]=strength, p[1]=angle(radians)
            let s = slots[i].params[0];
            let ang = slots[i].params[1];
            let dir = vec2f(cos(ang), sin(ang));
            r_off = dir * s;
            b_off = -dir * s;
            break; // 同一帧只处理一个色差 effect
        }
    }

    var out: VertexOutput;
    out.clip_pos = vec4f(pos, 0.0, 1.0);
    out.uv = uv;
    out.ab_r_offset = r_off;
    out.ab_b_offset = b_off;
    return out;
}
```

### 9.4 Fragment Shader 色差处理

色差需要在 Vertex Shader 和 Fragment Shader 两端联动——VS 写入 UV 偏移量，FS 分别对 R/G/B 通道采样不同 UV：

```wgsl
@group(0) @binding(1) var t_source: texture_2d<f32>;
@group(0) @binding(2) var s_sampler: sampler;

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r_uv = clamp(in.uv + in.ab_r_offset, vec2f(0.0), vec2f(1.0));
    let b_uv = clamp(in.uv + in.ab_b_offset, vec2f(0.0), vec2f(1.0));
    let r = textureSample(t_source, s_sampler, r_uv).r;
    let g = textureSample(t_source, s_sampler, in.uv).g;
    let b = textureSample(t_source, s_sampler, b_uv).b;
    let a = textureSample(t_source, s_sampler, in.uv).a;
    return vec4f(r, g, b, a);
}
```

### 9.5 API 设计

#### 9.5.1 Effect Type 注册（扩展点）

```c
/**
 * 注册一个新的 effect type。
 * 在 effect slot 中首次使用前调用一次（内部缓存元数据）。
 * 每次 encode 前调用一次（内部缓存 shader 中的路由表索引）。
 */
typedef void (*FS_PPEffectValidatorFn)(
    const float params[FS_PP_MAX_PARAM_F32]
);

uint32_t fs_pp_register_effect_type(
    FS_PPEffectType type,
    const char* effect_name,
    FS_PPEffectValidatorFn validator  // 可选，参数范围检查
);

/**
 * 查询已注册的 effect type。
 * @return type slot index，-1 表示未注册
 */
int32_t fs_pp_find_effect_type(const char* effect_name);
```

#### 9.5.2 Slot 管理

```c
/**
 * 启用一个 effect slot。
 * @param slot  slot 索引 [0, FS_PP_MAX_EFFECTS)
 * @param type  effect type
 * @return      true 成功，false slot 超出范围或 type 未注册
 */
bool fs_pp_effect_enable(
    FS_Core* core,
    uint32_t slot,
    FS_PPEffectType type
);

/**
 * 禁用一个 effect slot。
 */
void fs_pp_effect_disable(FS_Core* core, uint32_t slot);

/**
 * 查询 slot 是否激活。
 */
bool fs_pp_effect_is_enabled(const FS_Core* core, uint32_t slot);

/**
 * 重置所有 slot（每帧开始前调用）。
 */
void fs_pp_clear_all_effects(FS_Core* core);
```

#### 9.5.3 参数设置

```c
/**
 * 设置 float 参数。
 * @param slot   effect slot
 * @param param  参数索引 [0, FS_PP_MAX_PARAM_F32)
 * @param value  参数值
 */
bool fs_pp_effect_set_f32(
    FS_Core* core, uint32_t slot,
    uint32_t param, float value
);

/**
 * 批量设置参数（从数组复制）。
 */
bool fs_pp_effect_set_f32_array(
    FS_Core* core, uint32_t slot,
    const float values[], uint32_t count
);

/**
 * 设置叠加权重。
 */
bool fs_pp_effect_set_weight(FS_Core* core, uint32_t slot, float weight);

/**
 * 批量设置参数（Variant，支持 vec2/vec4）。
 */
bool fs_pp_effect_set_vec2(
    FS_Core* core, uint32_t slot,
    uint32_t param_index, float x, float y
);
bool fs_pp_effect_set_vec4(
    FS_Core* core, uint32_t slot,
    uint32_t param_index, float x, float y, float z, float w
);
```

#### 9.5.4 鼠标/触摸状态

```c
/**
 * 更新鼠标/触摸状态。每帧调用一次。
 * 内部自动将像素坐标转换为 UV 坐标传入 shader。
 *
 * @param x, y         当前指针位置（像素，左上原点）
 * @param click_x, y   最近一次按下时的位置（像素，左上原点）
 * @param down          当前是否按下
 */
void fs_pp_set_mouse(
    FS_Core* core,
    float x, float y,
    float click_x, float click_y,
    bool down
);

/**
 * 查询当前鼠标状态。
 */
bool fs_pp_get_mouse(const FS_Core* core,
    float* out_x, float* out_y,
    float* out_click_x, float* out_click_y,
    bool* out_down);
```

#### 9.5.5 使用示例

```c
// === 初始化时：注册所有内置 effect ===
fs_pp_register_effect_type(FS_PP_EFFECT_RIPPLE,       "ripple",       NULL);
fs_pp_register_effect_type(FS_PP_EFFECT_BULGE,        "bulge",        NULL);
fs_pp_register_effect_type(FS_PP_EFFECT_WAVE,          "wave",         NULL);
fs_pp_register_effect_type(FS_PP_EFFECT_SWIRL,        "swirl",        NULL);
fs_pp_register_effect_type(FS_PP_EFFECT_CHROMATIC_AB, "chromatic_ab", NULL);
fs_pp_register_effect_type(FS_PP_EFFECT_PERSPECTIVE,   "perspective",  NULL);

// === 每帧 render loop ===

// 鼠标事件处理
void on_mouse_move(float x, float y) {
    fs_pp_set_mouse(core, x, y,
        core->last_click_x, core->last_click_y,
        core->mouse_down);
}
void on_mouse_down(float x, float y) {
    core->last_click_x = x; core->last_click_y = y;
    core->mouse_down = true;
    fs_pp_set_mouse(core, x, y, x, y, true);
}
void on_mouse_up() {
    core->mouse_down = false;
}

// 每帧开始前重置所有 slot
fs_pp_clear_all_effects(core);

// --- 设置 Slot 0: 鼠标触发的水波涟漪 ---
fs_pp_effect_enable(core, 0, FS_PP_EFFECT_RIPPLE);
// 鼠标当前位置转换为 UV
fs_pp_effect_set_vec2(core, 0, 0,
    mouse_x / core->width, mouse_y / core->height);  // center
fs_pp_effect_set_f32(core, 0, 2, 0.03f);   // amplitude
fs_pp_effect_set_f32(core, 0, 3, 8.0f);    // frequency
fs_pp_effect_set_f32(core, 0, 4, 5.0f);    // speed
fs_pp_effect_set_f32(core, 0, 5, 3.0f);    // decay
fs_pp_effect_set_weight(core, 0, 1.0f);

// --- 设置 Slot 1: 静态凸镜（叠加） ---
fs_pp_effect_enable(core, 1, FS_PP_EFFECT_BULGE);
fs_pp_effect_set_vec2(core, 1, 0, 0.5f, 0.5f); // center
fs_pp_effect_set_f32(core, 1, 2, 0.3f);          // radius
fs_pp_effect_set_f32(core, 1, 3, 0.5f);          // strength (+=凸)
fs_pp_effect_set_weight(core, 1, 0.7f);

// --- 设置 Slot 2: 色差效果 ---
fs_pp_effect_enable(core, 2, FS_PP_EFFECT_CHROMATIC_AB);
fs_pp_effect_set_f32(core, 2, 0, 0.02f);   // strength
fs_pp_effect_set_f32(core, 2, 1, 0.785f);   // angle = PI/4

// === postprocess_encode 自动读取 uniforms 并应用所有顶点变形 ===
```

### 9.6 扩展性：添加新 Effect 的完整流程

添加一个全新 effect（如 fisheye 鱼眼效果）只需以下步骤：

**Step 1：C 端枚举**

```c
// 在 FS_PPEffectType 枚举中添加：
typedef enum FS_PPEffectType {
    FS_PP_EFFECT_NONE           = 0,
    FS_PP_EFFECT_RIPPLE         = 1,
    FS_PP_EFFECT_BULGE          = 2,
    FS_PP_EFFECT_WAVE           = 3,
    FS_PP_EFFECT_SWIRL          = 4,
    FS_PP_EFFECT_CHROMATIC_AB   = 5,
    FS_PP_EFFECT_PERSPECTIVE    = 6,
    FS_PP_EFFECT_FISHEYE        = 7,  // ← 新增
    FS_PP_EFFECT_CUSTOM_0       = 64,
} FS_PPEffectType;
```

**Step 2：WGSL 端实现**

```wgsl
// 在 fullstack_postprocess_vertex.wgsl 中：
fn effect_fisheye(pos: vec2f, uv: vec2f, p: const f32) -> vec2f {
    // p[0]=strength, p[1]=zoom
    let d = length(uv - 0.5) * 2.0;
    let z = sqrt(max(0.0, 1.0 - d*d)) + p[0];
    let r = atan2(d, z) / 3.14159 * p[1];
    let theta = atan2(uv.y - 0.5, uv.x - 0.5);
    let fisheye_uv = vec2f(cos(theta), sin(theta)) * r + 0.5;
    // 注意：fishye 需要片段级 UV 采样，单独用一个 pass 处理
    return pos;
}
```

**Step 3：WGSL 路由更新**

```wgsl
switch (slots[i].type) {
    case 1u:  deformed = effect_ripple(pos, uv, p);    break;
    case 2u:  deformed = effect_bulge(pos, uv, p);     break;
    case 3u:  deformed = effect_wave(pos, uv, p);      break;
    case 4u:  deformed = effect_swirl(pos, uv, p);      break;
    case 6u:  deformed = effect_perspective(pos, uv, p); break;
    case 7u:  deformed = effect_fisheye(pos, uv, p);   break;  // ← 新增
    default: break;
}
```

**Step 4：C 端注册**

```c
fs_pp_register_effect_type(FS_PP_EFFECT_FISHEYE, "fisheye", NULL);
```

**Step 5：使用**

```c
fs_pp_effect_enable(core, 3, FS_PP_EFFECT_FISHEYE);
fs_pp_effect_set_f32(core, 3, 0, 0.5f);  // strength
fs_pp_effect_set_f32(core, 3, 1, 1.2f);  // zoom
```

### 9.7 UI 交互数据流

```
用户鼠标/触摸输入
    ↓
C 端事件回调 (on_mouse_move / on_mouse_down / on_mouse_up)
    ↓
fs_pp_set_mouse(core, x, y, click_x, click_y, down)
    ↓
fs_postprocess_encode()
    ├── fs_pp_upload_uniforms()   ← mouse 数据写入 uniform buffer
    ├── wgpuQueueWriteBuffer()     ← 上传到 GPU
    └── vertex shader 执行
           ↓
    apply_vertex_deform() 路由到 effect_XXX()
           ↓
    鼠标位置驱动 params → 顶点变形 → 水波/凸镜/色差动画
           ↓
    画面实时响应
```

### 9.8 性能说明

- Vertex Shader 中的 effect 函数极其轻量（sin/cos/dot/distance 等向量运算）
- effect_count=0 时，vertex shader 跳过整个 deform 循环，走固定全屏四边形路径
- 每帧额外开销：uniform buffer 上传（240 字节）+ Vertex Shader 若干条向量指令
- 即使同时激活 8 个 effect，额外 GPU 指令数也极少（<100 条/顶点），对帧率影响可忽略

### 9.9 与 Fragment Shader Effect 的关系

本文档中的顶点交互方案与原报告第 5 节的 Fragment Shader effect（如 blur、bloom、vignette）**完全正交**、可以叠加：

- **Vertex Shader effect**：改变像素在屏幕上的位置（空间变形）
- **Fragment Shader effect**：改变像素的颜色值（颜色处理）

实际渲染时，vertex deform 先执行（改变 UV 采样位置），fragment pass 后执行（改变采样后的颜色值）。两者独立设计、统一通过 uniform params 控制。
