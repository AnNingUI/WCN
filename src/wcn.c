#define USE_MATH_DEFINES
#include "WCN/WCN.h"
#include "WCN/WCN_Math.h"
#include "shader_manager.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 绘图状态结构体
typedef struct {
  WCN_Color fill_color;
  WCN_Color stroke_color;
  float line_width;
  // 可以添加更多状态如变换矩阵等
} WCN_DrawState;

// 路径操作枚举
typedef enum {
  WCN_PATH_MOVE_TO,
  WCN_PATH_LINE_TO,
  WCN_PATH_BEZIER_CURVE_TO,
  WCN_PATH_QUADRATIC_CURVE_TO,
  WCN_PATH_ARC,
  WCN_PATH_CLOSE
} WCN_PathOpType;

// 路径操作结构体
typedef struct {
  WCN_PathOpType type;
  union {
    WCN_Point point;
    struct {
      WCN_Point cp1;
      WCN_Point cp2;
      WCN_Point end;
    } bezier;
    struct {
      WCN_Point cp;
      WCN_Point end;
    } quadratic;
    struct {
      WCN_Point center;
      float radius;
      float startAngle;
      float endAngle;
      bool anticlockwise;
    } arc;
  } data;
} WCN_PathOp;

// 变换矩阵类型别名 - 使用 wcn_math.c 的 Mat3
typedef WMATH_TYPE(Mat3) WCN_TransformMatrix;

// 上下文结构体
struct WCN_Context {
  // 可以添加全局状态信息
  bool initialized;
  WCN_ShaderManager *shader_manager; // 修复类型声明
};

// Canvas 结构体
struct WCN_Canvas {
  WGPUDevice device;
  WGPUTextureFormat format;
  WGPUTextureView view;
  WGPUCommandEncoder encoder;
  WGPURenderPassEncoder pass;
  WCN_ShaderManager *shader_manager; // 修复类型声明
  uint32_t width;
  uint32_t height;
  WCN_Color fill_color;
  WCN_Color stroke_color;
  float line_width;

  // 路径相关
  WCN_PathOp *path_ops;
  size_t path_ops_count;
  size_t path_ops_capacity;

  // 绘图状态栈
  WCN_DrawState *state_stack;
  size_t state_stack_count;
  size_t state_stack_capacity;

  // 顶点缓冲区
  WGPUBuffer vertex_buffer;
  uint32_t vertex_count;
  uint32_t vertex_capacity;

  // Uniform缓冲区
  WGPUBuffer uniform_buffer;

  // 变换矩阵栈
  WCN_TransformMatrix *transform_stack;
  size_t transform_stack_count;
  size_t transform_stack_capacity;
  WCN_TransformMatrix current_transform;

  // Bind group
  WGPUBindGroup bind_group;

  // 批渲染相关
  WCN_Vertex *batch_vertices;
  uint32_t batch_vertex_count;
  uint32_t batch_vertex_capacity;
};

// 初始化上下文
WCN_Context *wcn_init_context(void) {
  WCN_Context *context = (WCN_Context *)malloc(sizeof(WCN_Context));
  if (context) {
    context->initialized = true;
    context->shader_manager = NULL;
  }
  return context;
}

// 设置上下文的设备（用于初始化着色器管理器）
void wcn_context_set_device(WCN_Context *context, WGPUDevice device) {
  if (context && device) {
    context->shader_manager = wcn_shader_manager_init(device);
  }
}

// 销毁上下文
void wcn_destroy_context(WCN_Context *context) {
  if (context) {
    if (context->shader_manager) {
      wcn_shader_manager_destroy(context->shader_manager);
    }
    free(context);
  }
}

// 创建 Canvas
WCN_Canvas *wcn_create_canvas(WCN_Context *context, WGPUDevice device,
                              WGPUTextureFormat format, uint32_t width,
                              uint32_t height) {
  if (!context || !context->initialized || !device) {
    return NULL;
  }

  // 如果上下文还没有着色器管理器，初始化它
  if (!context->shader_manager) {
    wcn_context_set_device(context, device);
  }

  WCN_Canvas *canvas = (WCN_Canvas *)malloc(sizeof(WCN_Canvas));
  if (!canvas) {
    return NULL;
  }

  canvas->device = device;
  canvas->format = format;
  canvas->width = width;
  canvas->height = height;
  canvas->shader_manager = context->shader_manager;
  canvas->fill_color = (WCN_Color){1.0f, 1.0f, 1.0f, 1.0f};   // 默认白色
  canvas->stroke_color = (WCN_Color){0.0f, 0.0f, 0.0f, 1.0f}; // 默认黑色
  canvas->line_width = 1.0f;
  canvas->vertex_buffer = NULL;
  canvas->uniform_buffer = NULL;
  canvas->vertex_count = 0;
  canvas->bind_group = NULL;

  // 初始化路径操作数组
  canvas->path_ops_capacity = 16;
  canvas->path_ops_count = 0;
  canvas->path_ops =
      (WCN_PathOp *)malloc(canvas->path_ops_capacity * sizeof(WCN_PathOp));

  // 初始化状态栈
  canvas->state_stack_capacity = 8;
  canvas->state_stack_count = 0;
  canvas->state_stack = (WCN_DrawState *)malloc(canvas->state_stack_capacity *
                                                sizeof(WCN_DrawState));

  // 初始化变换矩阵栈
  canvas->transform_stack_capacity = 8;
  canvas->transform_stack_count = 0;
  canvas->transform_stack = (WCN_TransformMatrix *)malloc(
      canvas->transform_stack_capacity * sizeof(WCN_TransformMatrix));

  // 初始化当前变换矩阵为单位矩阵
  canvas->current_transform = WMATH_IDENTITY(Mat3)();

  // 初始化批渲染数据
  canvas->batch_vertex_capacity = 1024; // 初始容量
  canvas->batch_vertex_count = 0;
  canvas->batch_vertices =
      (WCN_Vertex *)malloc(canvas->batch_vertex_capacity * sizeof(WCN_Vertex));

  // 创建纹理视图（这里需要从外部传入纹理视图）
  canvas->view = NULL;
  canvas->encoder = NULL;
  canvas->pass = NULL;

  // 创建uniform缓冲区
  canvas->uniform_buffer = wgpuDeviceCreateBuffer(
      canvas->device,
      &(WGPUBufferDescriptor){
          .label = "Uniform Buffer",
          .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
          .size = 16, // vec2<f32> resolution (aligned to 16 bytes)
          .mappedAtCreation = false,
      });

  // 更新uniform缓冲区数据
  float uniform_data[4]; // 4 for aligned resolution (vec2<f32> aligned to 16
                         // bytes)
  uniform_data[0] = (float)width;
  uniform_data[1] = (float)height;
  uniform_data[2] = 0.0f; // padding
  uniform_data[3] = 0.0f; // padding

  wgpuQueueWriteBuffer(wgpuDeviceGetQueue(canvas->device),
                       canvas->uniform_buffer, 0, uniform_data,
                       4 * sizeof(float));

  return canvas;
}

// 设置纹理视图
void wcn_canvas_set_texture_view(WCN_Canvas *canvas, WGPUTextureView view) {
  if (canvas && view) {
    canvas->view = view;
  }
}

// 更新Canvas尺寸
void wcn_canvas_set_size(WCN_Canvas *canvas, uint32_t width, uint32_t height) {
  if (!canvas) {
    return;
  }

  canvas->width = width;
  canvas->height = height;

  // 更新uniform缓冲区数据
  if (canvas->uniform_buffer) {
    float uniform_data[4]; // 4 for aligned resolution (vec2<f32> aligned to 16
                           // bytes)
    uniform_data[0] = (float)width;
    uniform_data[1] = (float)height;
    uniform_data[2] = 0.0f; // padding
    uniform_data[3] = 0.0f; // padding

    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(canvas->device),
                         canvas->uniform_buffer, 0, uniform_data,
                         4 * sizeof(float));
  }
}

// 销毁 Canvas
void wcn_destroy_canvas(WCN_Canvas *canvas) {
  if (canvas) {
    if (canvas->pass) {
      wgpuRenderPassEncoderEnd(canvas->pass);
      wgpuRenderPassEncoderRelease(canvas->pass);
    }
    if (canvas->encoder) {
      wgpuCommandEncoderRelease(canvas->encoder);
    }
    if (canvas->view) {
      wgpuTextureViewRelease(canvas->view);
    }

    // 释放顶点缓冲区
    if (canvas->vertex_buffer) {
      wgpuBufferRelease(canvas->vertex_buffer);
    }

    // 释放uniform缓冲区
    if (canvas->uniform_buffer) {
      wgpuBufferRelease(canvas->uniform_buffer);
    }

    // 释放bind group
    if (canvas->bind_group) {
      wgpuBindGroupRelease(canvas->bind_group);
    }

    // 释放路径操作数组
    if (canvas->path_ops) {
      free(canvas->path_ops);
    }

    // 释放状态栈
    if (canvas->state_stack) {
      free(canvas->state_stack);
    }

    // 释放变换矩阵栈
    if (canvas->transform_stack) {
      free(canvas->transform_stack);
    }

    // 释放批渲染数据
    if (canvas->batch_vertices) {
      free(canvas->batch_vertices);
    }

    free(canvas);
  }
}

// 创建渲染管线
static WGPURenderPipeline create_render_pipeline(WCN_Canvas *canvas) {
  if (!canvas || !canvas->shader_manager) {
    WCN_DEBUG_PRINT("Invalid canvas or shader manager");
    return NULL;
  }

  WCN_ShaderModule shader_module =
      wcn_shader_manager_get_color_shader(canvas->shader_manager);

  // 检查着色器模块是否有效
  if (!shader_module.vertex_shader || !shader_module.fragment_shader) {
    WCN_DEBUG_PRINT("Invalid shader modules");
    return NULL;
  }

  // 顶点属性
  WGPUVertexAttribute vertex_attributes[2] = {
      (WGPUVertexAttribute){
          .format = WGPUVertexFormat_Float32x2,
          .offset = 0,
          .shaderLocation = 0,
      },
      (WGPUVertexAttribute){
          .format = WGPUVertexFormat_Float32x4,
          .offset = 2 * sizeof(float),
          .shaderLocation = 1,
      }};

  // 顶点缓冲区布局
  WGPUVertexBufferLayout vertex_buffer_layout = (WGPUVertexBufferLayout){
      .arrayStride = sizeof(WCN_Vertex),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 2,
      .attributes = vertex_attributes,
  };

  // 创建bind group布局
  WGPUBindGroupLayoutEntry bindGroupLayoutEntries[1] = {
      (WGPUBindGroupLayoutEntry){
          .binding = 0,
          .visibility = WGPUShaderStage_Vertex,
          .buffer =
              (WGPUBufferBindingLayout){
                  .type = WGPUBufferBindingType_Uniform,
                  .minBindingSize = 4 * sizeof(float), // resolution only
              },
      }};

  WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {
      .entryCount = 1,
      .entries = bindGroupLayoutEntries,
  };

  WGPUBindGroupLayout bindGroupLayout =
      wgpuDeviceCreateBindGroupLayout(canvas->device, &bindGroupLayoutDesc);

  // 管线布局
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
      canvas->device, &(WGPUPipelineLayoutDescriptor){
                          .bindGroupLayoutCount = 1,
                          .bindGroupLayouts = &bindGroupLayout,
                      });

  if (!pipeline_layout) {
    WCN_DEBUG_PRINT("Failed to create pipeline layout");
    wgpuBindGroupLayoutRelease(bindGroupLayout);
    return NULL;
  }

  // 设置正确的入口点名称
  WGPUStringView vertex_entry_point = {.data = "main", .length = 4};
  WGPUStringView fragment_entry_point = {.data = "main", .length = 4};

  // 渲染管线描述符
  WGPURenderPipelineDescriptor pipeline_descriptor =
      (WGPURenderPipelineDescriptor){
          .layout = pipeline_layout,
          .vertex =
              (WGPUVertexState){
                  .module = shader_module.vertex_shader,
                  .entryPoint = vertex_entry_point, // 使用正确的入口点
                  .bufferCount = 1,
                  .buffers = &vertex_buffer_layout,
              },
          .primitive =
              (WGPUPrimitiveState){
                  .topology = WGPUPrimitiveTopology_TriangleList,
                  .stripIndexFormat = WGPUIndexFormat_Undefined,
                  .frontFace = WGPUFrontFace_CCW,
                  .cullMode = WGPUCullMode_None,
              },
          .multisample =
              (WGPUMultisampleState){
                  .count = 1,
                  .mask = ~0u,
                  .alphaToCoverageEnabled = false,
              },
          .fragment =
              &(WGPUFragmentState){
                  .module = shader_module.fragment_shader,
                  .entryPoint = fragment_entry_point, // 使用正确的入口点
                  .targetCount = 1,
                  .targets =
                      &(WGPUColorTargetState){
                          .format = canvas->format,
                          .blend =
                              &(WGPUBlendState){
                                  .color =
                                      (WGPUBlendComponent){
                                          .operation = WGPUBlendOperation_Add,
                                          .srcFactor = WGPUBlendFactor_SrcAlpha,
                                          .dstFactor =
                                              WGPUBlendFactor_OneMinusSrcAlpha,
                                      },
                                  .alpha =
                                      (WGPUBlendComponent){
                                          .operation = WGPUBlendOperation_Add,
                                          .srcFactor = WGPUBlendFactor_One,
                                          .dstFactor =
                                              WGPUBlendFactor_OneMinusSrcAlpha,
                                      },
                              },
                          .writeMask = WGPUColorWriteMask_All,
                      },
              },
      };

  const WGPURenderPipeline pipeline =
      wgpuDeviceCreateRenderPipeline(canvas->device, &pipeline_descriptor);

  // 检查渲染管线是否创建成功
  if (!pipeline) {
    WCN_DEBUG_PRINT("Failed to create render pipeline");
  } else {
    WCN_DEBUG_PRINT("Render pipeline created successfully");
  }

  wgpuPipelineLayoutRelease(pipeline_layout);
  wgpuBindGroupLayoutRelease(bindGroupLayout);

  return pipeline;
}

// 更新顶点缓冲区
static void update_vertex_buffer(WCN_Canvas *canvas, WCN_Vertex *vertices,
                                 uint32_t count) {
  if (!canvas || !vertices || count == 0) {
    return;
  }

  // 创建或更新顶点缓冲区
  if (!canvas->vertex_buffer) {
    canvas->vertex_buffer = wgpuDeviceCreateBuffer(
        canvas->device,
        &(WGPUBufferDescriptor){
            .label = "Vertex Buffer",
            .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
            .size = count * sizeof(WCN_Vertex),
            .mappedAtCreation = false,
        });
  } else if (canvas->vertex_count < count) {
    // 如果现有缓冲区太小，释放并重新创建
    wgpuBufferRelease(canvas->vertex_buffer);
    canvas->vertex_buffer = wgpuDeviceCreateBuffer(
        canvas->device,
        &(WGPUBufferDescriptor){
            .label = "Vertex Buffer",
            .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
            .size = count * sizeof(WCN_Vertex),
            .mappedAtCreation = false,
        });
  }

  // 更新顶点数据
  wgpuQueueWriteBuffer(wgpuDeviceGetQueue(canvas->device),
                       canvas->vertex_buffer, 0, vertices,
                       count * sizeof(WCN_Vertex));
  canvas->vertex_count = count;
}

// 确保批渲染顶点数组有足够的容量
static void ensure_batch_capacity(WCN_Canvas *canvas,
                                  uint32_t required_capacity) {
  if (canvas->batch_vertex_capacity < required_capacity) {
    uint32_t new_capacity = canvas->batch_vertex_capacity * 2;
    if (new_capacity < required_capacity) {
      new_capacity = required_capacity;
    }

    WCN_Vertex *new_vertices = (WCN_Vertex *)realloc(
        canvas->batch_vertices, new_capacity * sizeof(WCN_Vertex));
    if (new_vertices) {
      canvas->batch_vertices = new_vertices;
      canvas->batch_vertex_capacity = new_capacity;
    }
  }
}

// 绘制顶点（添加到批渲染缓冲区）
static void draw_vertices(WCN_Canvas *canvas, WCN_Vertex *vertices,
                          uint32_t count) {
  if (!canvas || !canvas->pass || !vertices || count == 0) {
    WCN_DEBUG_PRINT("Invalid parameters for draw_vertices");
    return;
  }

  // 确保批渲染缓冲区有足够的容量
  ensure_batch_capacity(canvas, canvas->batch_vertex_count + count);

  // 将顶点添加到批渲染缓冲区（不应用变换矩阵，变换将在着色器中处理）
  for (uint32_t i = 0; i < count; i++) {
    canvas->batch_vertices[canvas->batch_vertex_count + i] = vertices[i];
  }

  canvas->batch_vertex_count += count;
  WCN_DEBUG_PRINT("Added %d vertices to batch, total: %d", count,
                  canvas->batch_vertex_count);
}

// 批渲染所有顶点
void wcn_flush_batch(WCN_Canvas *canvas) {
  if (!canvas || !canvas->pass || canvas->batch_vertex_count == 0) {
    return;
  }

  WCN_DEBUG_PRINT("Flushing batch with %d vertices",
                  canvas->batch_vertex_count);

  // 更新顶点缓冲区
  update_vertex_buffer(canvas, canvas->batch_vertices,
                       canvas->batch_vertex_count);

  // 更新uniform缓冲区中的分辨率
  if (canvas->uniform_buffer) {
    float uniform_data[4]; // 4 for aligned resolution (vec2<f32> aligned to 16
                           // bytes)
    uniform_data[0] = (float)canvas->width;
    uniform_data[1] = (float)canvas->height;
    uniform_data[2] = 0.0f; // padding
    uniform_data[3] = 0.0f; // padding

    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(canvas->device),
                         canvas->uniform_buffer, 0, uniform_data,
                         4 * sizeof(float));
  }

  // 创建渲染管线（如果还没有创建）
  static WGPURenderPipeline pipeline = NULL;
  if (!pipeline) {
    pipeline = create_render_pipeline(canvas);
    if (!pipeline) {
      WCN_DEBUG_PRINT("Failed to create pipeline in wcn_flush_batch");
      return;
    }
    WCN_DEBUG_PRINT("Created render pipeline successfully");
  }

  // 设置渲染管线
  wgpuRenderPassEncoderSetPipeline(canvas->pass, pipeline);
  WCN_DEBUG_PRINT("Set pipeline for batch rendering");

  // 创建bind group（如果还没有创建）
  if (!canvas->bind_group) {
    // 创建bind group布局
    WGPUBindGroupLayoutEntry bindGroupLayoutEntries[1] = {
        (WGPUBindGroupLayoutEntry){
            .binding = 0,
            .visibility = WGPUShaderStage_Vertex,
            .buffer =
                (WGPUBufferBindingLayout){
                    .type = WGPUBufferBindingType_Uniform,
                    .minBindingSize = 4 * sizeof(float), // resolution only
                },
        }};

    WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {
        .entryCount = 1,
        .entries = bindGroupLayoutEntries,
    };

    WGPUBindGroupLayout bindGroupLayout =
        wgpuDeviceCreateBindGroupLayout(canvas->device, &bindGroupLayoutDesc);

    // 创建bind group
    WGPUBindGroupEntry bindGroupEntries[1] = {(WGPUBindGroupEntry){
        .binding = 0,
        .buffer = canvas->uniform_buffer,
        .offset = 0,
        .size = 4 * sizeof(float), // resolution only
    }};

    WGPUBindGroupDescriptor bindGroupDesc = {
        .layout = bindGroupLayout,
        .entryCount = 1,
        .entries = bindGroupEntries,
    };

    canvas->bind_group =
        wgpuDeviceCreateBindGroup(canvas->device, &bindGroupDesc);
    wgpuBindGroupLayoutRelease(bindGroupLayout);

    if (canvas->bind_group) {
      WCN_DEBUG_PRINT("Created bind group successfully");
    } else {
      WCN_DEBUG_PRINT("Failed to create bind group");
    }
  }

  // 设置bind group
  if (canvas->bind_group) {
    wgpuRenderPassEncoderSetBindGroup(canvas->pass, 0, canvas->bind_group, 0,
                                      NULL);
    WCN_DEBUG_PRINT("Set bind group");
  }

  // 设置顶点缓冲区
  wgpuRenderPassEncoderSetVertexBuffer(
      canvas->pass, 0, canvas->vertex_buffer, 0,
      canvas->batch_vertex_count * sizeof(WCN_Vertex));
  WCN_DEBUG_PRINT("Set vertex buffer with %d vertices",
                  canvas->batch_vertex_count);

  // 绘制
  wgpuRenderPassEncoderDraw(canvas->pass, canvas->batch_vertex_count, 1, 0, 0);
  WCN_DEBUG_PRINT("Drew %d vertices in batch", canvas->batch_vertex_count);

  // 重置批渲染计数器
  canvas->batch_vertex_count = 0;
}

// 开始渲染通道
void wcn_begin_render_pass(WCN_Canvas *canvas) {
  if (!canvas || !canvas->view) {
    return;
  }

  // 重置批渲染计数器
  canvas->batch_vertex_count = 0;

  // 创建命令编码器
  WGPUCommandEncoderDescriptor encoderDesc = {0};
  canvas->encoder =
      wgpuDeviceCreateCommandEncoder(canvas->device, &encoderDesc);

  // 创建渲染通道描述符，使用明显的清屏颜色（亮灰色）
  WGPURenderPassColorAttachment colorAttachment = {0};
  colorAttachment.view = canvas->view;
  colorAttachment.loadOp = WGPULoadOp_Clear;
  colorAttachment.storeOp = WGPUStoreOp_Store;
  colorAttachment.clearValue = (WGPUColor){0.2, 0.2, 0.2, 1.0}; // 深灰色背景

  WGPURenderPassDescriptor renderPassDesc = {0};
  renderPassDesc.colorAttachmentCount = 1;
  renderPassDesc.colorAttachments = &colorAttachment;

  // 开始渲染通道
  canvas->pass =
      wgpuCommandEncoderBeginRenderPass(canvas->encoder, &renderPassDesc);

  WCN_DEBUG_PRINT("Begin render pass - batch vertex count reset to 0");
}

// 结束渲染通道
void wcn_end_render_pass(WCN_Canvas *canvas) {
  if (!canvas || !canvas->pass) {
    return;
  }

  // 刷新批渲染缓冲区
  wcn_flush_batch(canvas);

  wgpuRenderPassEncoderEnd(canvas->pass);
  wgpuRenderPassEncoderRelease(canvas->pass);
  canvas->pass = NULL;
}

// 设置填充颜色
void wcn_set_fill_color(WCN_Canvas *canvas, WCN_Color color) {
  if (canvas) {
    canvas->fill_color = color;
  }
}

// 设置描边颜色
void wcn_set_stroke_color(WCN_Canvas *canvas, WCN_Color color) {
  if (canvas) {
    canvas->stroke_color = color;
  }
}

// 设置线条宽度
void wcn_set_line_width(WCN_Canvas *canvas, float width) {
  if (canvas && width > 0) {
    canvas->line_width = width;
  }
}

// 确保路径操作数组有足够的容量
static void ensure_path_ops_capacity(WCN_Canvas *canvas,
                                     size_t required_capacity) {
  if (canvas->path_ops_capacity < required_capacity) {
    size_t new_capacity = canvas->path_ops_capacity * 2;
    if (new_capacity < required_capacity) {
      new_capacity = required_capacity;
    }

    WCN_PathOp *new_ops = (WCN_PathOp *)realloc(
        canvas->path_ops, new_capacity * sizeof(WCN_PathOp));
    if (new_ops) {
      canvas->path_ops = new_ops;
      canvas->path_ops_capacity = new_capacity;
    }
  }
}

// 确保状态栈有足够的容量
static void ensure_state_stack_capacity(WCN_Canvas *canvas,
                                        size_t required_capacity) {
  if (canvas->state_stack_capacity < required_capacity) {
    size_t new_capacity = canvas->state_stack_capacity * 2;
    if (new_capacity < required_capacity) {
      new_capacity = required_capacity;
    }

    WCN_DrawState *new_stack = (WCN_DrawState *)realloc(
        canvas->state_stack, new_capacity * sizeof(WCN_DrawState));
    if (new_stack) {
      canvas->state_stack = new_stack;
      canvas->state_stack_capacity = new_capacity;
    }
  }
}

// 确保变换矩阵栈有足够的容量
static void ensure_transform_stack_capacity(WCN_Canvas *canvas,
                                            size_t required_capacity) {
  if (canvas->transform_stack_capacity < required_capacity) {
    size_t new_capacity = canvas->transform_stack_capacity * 2;
    if (new_capacity < required_capacity) {
      new_capacity = required_capacity;
    }

    WCN_TransformMatrix *new_stack = (WCN_TransformMatrix *)realloc(
        canvas->transform_stack, new_capacity * sizeof(WCN_TransformMatrix));
    if (new_stack) {
      canvas->transform_stack = new_stack;
      canvas->transform_stack_capacity = new_capacity;
    }
  }
}

// 保存当前绘图状态
void wcn_save(WCN_Canvas *canvas) {
  if (!canvas) {
    return;
  }

  // 保存绘图状态
  ensure_state_stack_capacity(canvas, canvas->state_stack_count + 1);

  WCN_DrawState state;
  state.fill_color = canvas->fill_color;
  state.stroke_color = canvas->stroke_color;
  state.line_width = canvas->line_width;

  canvas->state_stack[canvas->state_stack_count] = state;
  canvas->state_stack_count++;

  // 保存变换矩阵
  ensure_transform_stack_capacity(canvas, canvas->transform_stack_count + 1);
  canvas->transform_stack[canvas->transform_stack_count] =
      canvas->current_transform;
  canvas->transform_stack_count++;
}

// 恢复之前保存的绘图状态
void wcn_restore(WCN_Canvas *canvas) {
  if (!canvas || canvas->state_stack_count == 0 ||
      canvas->transform_stack_count == 0) {
    return;
  }

  // 恢复绘图状态
  canvas->state_stack_count--;
  WCN_DrawState state = canvas->state_stack[canvas->state_stack_count];

  canvas->fill_color = state.fill_color;
  canvas->stroke_color = state.stroke_color;
  canvas->line_width = state.line_width;

  // 恢复变换矩阵
  canvas->transform_stack_count--;
  canvas->current_transform =
      canvas->transform_stack[canvas->transform_stack_count];
}

// 填充矩形
void wcn_fill_rect(WCN_Canvas *canvas, float x, float y, float width,
                   float height) {
  if (!canvas || !canvas->pass) {
    WCN_DEBUG_PRINT("Invalid canvas or render pass in wcn_fill_rect");
    return;
  }

  WCN_DEBUG_PRINT("Filling rectangle: pos(%.1f, %.1f) size(%.1f x %.1f) "
                  "color(%.2f, %.2f, %.2f, %.2f)",
                  x, y, width, height, canvas->fill_color.r,
                  canvas->fill_color.g, canvas->fill_color.b,
                  canvas->fill_color.a);

  // 应用当前变换矩阵到矩形坐标
  const float x1 = x, y1 = y;
  const float x2 = x + width, y2 = y;
  const float x3 = x, y3 = y + height;
  const float x4 = x + width, y4 = y + height;

  // 应用变换矩阵
  const float tx1 = canvas->current_transform.m[0 * 4 + 0] * x1 +
                    canvas->current_transform.m[0 * 4 + 1] * y1 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty1 = canvas->current_transform.m[1 * 4 + 0] * x1 +
                    canvas->current_transform.m[1 * 4 + 1] * y1 +
                    canvas->current_transform.m[1 * 4 + 2];

  const float tx2 = canvas->current_transform.m[0 * 4 + 0] * x2 +
                    canvas->current_transform.m[0 * 4 + 1] * y2 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty2 = canvas->current_transform.m[1 * 4 + 0] * x2 +
                    canvas->current_transform.m[1 * 4 + 1] * y2 +
                    canvas->current_transform.m[1 * 4 + 2];

  const float tx3 = canvas->current_transform.m[0 * 4 + 0] * x3 +
                    canvas->current_transform.m[0 * 4 + 1] * y3 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty3 = canvas->current_transform.m[1 * 4 + 0] * x3 +
                    canvas->current_transform.m[1 * 4 + 1] * y3 +
                    canvas->current_transform.m[1 * 4 + 2];

  const float tx4 = canvas->current_transform.m[0 * 4 + 0] * x4 +
                    canvas->current_transform.m[0 * 4 + 1] * y4 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty4 = canvas->current_transform.m[1 * 4 + 0] * x4 +
                    canvas->current_transform.m[1 * 4 + 1] * y4 +
                    canvas->current_transform.m[1 * 4 + 2];

  // 创建两个三角形的顶点来组成矩形
  WCN_Vertex vertices[6];

  // 第一个三角形
  vertices[0] = (WCN_Vertex){{tx1, ty1},
                             {canvas->fill_color.r, canvas->fill_color.g,
                              canvas->fill_color.b, canvas->fill_color.a}};
  vertices[1] = (WCN_Vertex){{tx2, ty2},
                             {canvas->fill_color.r, canvas->fill_color.g,
                              canvas->fill_color.b, canvas->fill_color.a}};
  vertices[2] = (WCN_Vertex){{tx3, ty3},
                             {canvas->fill_color.r, canvas->fill_color.g,
                              canvas->fill_color.b, canvas->fill_color.a}};

  // 第二个三角形
  vertices[3] = (WCN_Vertex){{tx2, ty2},
                             {canvas->fill_color.r, canvas->fill_color.g,
                              canvas->fill_color.b, canvas->fill_color.a}};
  vertices[4] = (WCN_Vertex){{tx4, ty4},
                             {canvas->fill_color.r, canvas->fill_color.g,
                              canvas->fill_color.b, canvas->fill_color.a}};
  vertices[5] = (WCN_Vertex){{tx3, ty3},
                             {canvas->fill_color.r, canvas->fill_color.g,
                              canvas->fill_color.b, canvas->fill_color.a}};

  // 绘制顶点
  draw_vertices(canvas, vertices, 6);
}

// 辅助函数：创建线段的三角形化矩形
static void create_line_triangles(WCN_Vertex *vertices, float x1, float y1,
                                  float x2, float y2, float line_width,
                                  WCN_Color color) {
  // 计算线段的方向向量
  float dx = x2 - x1;
  float dy = y2 - y1;
  float len = sqrtf(dx * dx + dy * dy);

  if (len < 0.001f)
    return; // 线段太短，跳过

  // 归一化方向向量
  dx /= len;
  dy /= len;

  // 计算垂直向量（用于线宽）
  float px = -dy * line_width * 0.5f;
  float py = dx * line_width * 0.5f;

  // 创建矩形的四个顶点（两个三角形）
  // 第一个三角形
  vertices[0] =
      (WCN_Vertex){{x1 - px, y1 - py}, {color.r, color.g, color.b, color.a}};
  vertices[1] =
      (WCN_Vertex){{x1 + px, y1 + py}, {color.r, color.g, color.b, color.a}};
  vertices[2] =
      (WCN_Vertex){{x2 - px, y2 - py}, {color.r, color.g, color.b, color.a}};

  // 第二个三角形
  vertices[3] =
      (WCN_Vertex){{x1 + px, y1 + py}, {color.r, color.g, color.b, color.a}};
  vertices[4] =
      (WCN_Vertex){{x2 + px, y2 + py}, {color.r, color.g, color.b, color.a}};
  vertices[5] =
      (WCN_Vertex){{x2 - px, y2 - py}, {color.r, color.g, color.b, color.a}};
}

// 描边矩形
void wcn_stroke_rect(WCN_Canvas *canvas, float x, float y, float width,
                     float height) {
  if (!canvas || !canvas->pass) {
    return;
  }

  // 应用当前变换矩阵到矩形坐标
  const float x1 = x, y1 = y;
  const float x2 = x + width, y2 = y;
  const float x3 = x + width, y3 = y + height;
  const float x4 = x, y4 = y + height;

  // 应用变换矩阵
  const float tx1 = canvas->current_transform.m[0 * 4 + 0] * x1 +
                    canvas->current_transform.m[0 * 4 + 1] * y1 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty1 = canvas->current_transform.m[1 * 4 + 0] * x1 +
                    canvas->current_transform.m[1 * 4 + 1] * y1 +
                    canvas->current_transform.m[1 * 4 + 2];

  const float tx2 = canvas->current_transform.m[0 * 4 + 0] * x2 +
                    canvas->current_transform.m[0 * 4 + 1] * y2 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty2 = canvas->current_transform.m[1 * 4 + 0] * x2 +
                    canvas->current_transform.m[1 * 4 + 1] * y2 +
                    canvas->current_transform.m[1 * 4 + 2];

  const float tx3 = canvas->current_transform.m[0 * 4 + 0] * x3 +
                    canvas->current_transform.m[0 * 4 + 1] * y3 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty3 = canvas->current_transform.m[1 * 4 + 0] * x3 +
                    canvas->current_transform.m[1 * 4 + 1] * y3 +
                    canvas->current_transform.m[1 * 4 + 2];

  const float tx4 = canvas->current_transform.m[0 * 4 + 0] * x4 +
                    canvas->current_transform.m[0 * 4 + 1] * y4 +
                    canvas->current_transform.m[0 * 4 + 2];
  const float ty4 = canvas->current_transform.m[1 * 4 + 0] * x4 +
                    canvas->current_transform.m[1 * 4 + 1] * y4 +
                    canvas->current_transform.m[1 * 4 + 2];

  // 创建矩形边框的顶点（每条边使用2个三角形，共24个顶点）
  WCN_Vertex vertices[24]; // 4条边 * 6个顶点（2个三角形）

  // 上边
  create_line_triangles(&vertices[0], tx1, ty1, tx2, ty2, canvas->line_width,
                        canvas->stroke_color);

  // 右边
  create_line_triangles(&vertices[6], tx2, ty2, tx3, ty3, canvas->line_width,
                        canvas->stroke_color);

  // 下边
  create_line_triangles(&vertices[12], tx3, ty3, tx4, ty4, canvas->line_width,
                        canvas->stroke_color);

  // 左边
  create_line_triangles(&vertices[18], tx4, ty4, tx1, ty1, canvas->line_width,
                        canvas->stroke_color);

  // 绘制顶点
  draw_vertices(canvas, vertices, 24);
}

// 辅助函数：计算贝塞尔曲线上的点
static WCN_Point bezier_point(WCN_Point p0, WCN_Point p1, WCN_Point p2,
                              WCN_Point p3, float t) {
  const float u = 1.0f - t;
  const float tt = t * t;
  const float uu = u * u;
  const float uuu = uu * u;
  const float ttt = tt * t;

  WCN_Point point;
  point.x = uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x;
  point.y = uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y;

  return point;
}

// 辅助函数：计算二次贝塞尔曲线上的点
static WCN_Point quadratic_bezier_point(WCN_Point p0, WCN_Point p1,
                                        WCN_Point p2, float t) {
  const float u = 1.0f - t;
  const float uu = u * u;
  const float tt = t * t;

  WCN_Point point;
  point.x = uu * p0.x + 2 * u * t * p1.x + tt * p2.x;
  point.y = uu * p0.y + 2 * u * t * p1.y + tt * p2.y;

  return point;
}

// 辅助函数：计算弧线上的点
static WCN_Point arc_point(WCN_Point center, float radius, float angle) {
  WCN_Point point;
  point.x = center.x + radius * cosf(angle);
  point.y = center.y + radius * sinf(angle);
  return point;
}

// 填充路径
void wcn_fill_path(WCN_Canvas *canvas) {
  if (!canvas || !canvas->pass || canvas->path_ops_count == 0) {
    return;
  }

  // 将路径转换为点序列
  const size_t max_points = canvas->path_ops_count * 32; // 为曲线预留更多点
  WCN_Point *points = (WCN_Point *)malloc(max_points * sizeof(WCN_Point));
  size_t point_count = 0;

  WCN_Point current_pos = {0, 0};
  WCN_Point path_start = {0, 0};
  bool path_started = false;

  for (size_t i = 0; i < canvas->path_ops_count && point_count < max_points;
       i++) {
    switch (canvas->path_ops[i].type) {
    case WCN_PATH_MOVE_TO:
      current_pos = canvas->path_ops[i].data.point;
      path_start = current_pos;
      path_started = true;
      if (point_count < max_points) {
        points[point_count++] = current_pos;
      }
      break;

    case WCN_PATH_LINE_TO:
      current_pos = canvas->path_ops[i].data.point;
      if (point_count < max_points) {
        points[point_count++] = current_pos;
      }
      break;

    case WCN_PATH_BEZIER_CURVE_TO: {
      const WCN_Point p0 = current_pos;
      const WCN_Point p1 = canvas->path_ops[i].data.bezier.cp1;
      const WCN_Point p2 = canvas->path_ops[i].data.bezier.cp2;
      const WCN_Point p3 = canvas->path_ops[i].data.bezier.end;

      // 将贝塞尔曲线分割为多个线段
      const int segments = 16;
      for (int j = 1; j <= segments && point_count < max_points; j++) {
        const float t = (float)j / (float)segments;
        points[point_count++] = bezier_point(p0, p1, p2, p3, t);
      }
      current_pos = p3;
      break;
    }

    case WCN_PATH_QUADRATIC_CURVE_TO: {
      const WCN_Point p0 = current_pos;
      const WCN_Point p1 = canvas->path_ops[i].data.quadratic.cp;
      const WCN_Point p2 = canvas->path_ops[i].data.quadratic.end;

      // 将二次贝塞尔曲线分割为多个线段
      const int segments = 16;
      for (int j = 1; j <= segments && point_count < max_points; j++) {
        const float t = (float)j / (float)segments;
        points[point_count++] = quadratic_bezier_point(p0, p1, p2, t);
      }
      current_pos = p2;
      break;
    }

    case WCN_PATH_ARC: {
      const WCN_Point center = canvas->path_ops[i].data.arc.center;
      const float radius = canvas->path_ops[i].data.arc.radius;
      const float start_angle = canvas->path_ops[i].data.arc.startAngle;
      const float end_angle = canvas->path_ops[i].data.arc.endAngle;
      const bool anticlockwise = canvas->path_ops[i].data.arc.anticlockwise;

      // 计算角度差
      float angle_diff = end_angle - start_angle;
      if (anticlockwise && angle_diff > 0) {
        angle_diff -= 2 * M_PI;
      } else if (!anticlockwise && angle_diff < 0) {
        angle_diff += 2 * M_PI;
      }

      // 将弧线分割为多个线段
      int segments = (int)(fabsf(angle_diff) * 16 / M_PI);
      if (segments < 4)
        segments = 4;
      if (segments > 64)
        segments = 64;

      for (int j = 1; j <= segments && point_count < max_points; j++) {
        float t = (float)j / segments;
        float angle = start_angle + t * angle_diff;
        points[point_count++] = arc_point(center, radius, angle);
      }
      current_pos = arc_point(center, radius, end_angle);
      break;
    }

    case WCN_PATH_CLOSE:
      if (path_started && point_count < max_points) {
        points[point_count++] = path_start;
      }
      break;
    }
  }

  // 使用三角形扇形填充多边形
  if (point_count >= 3) {
    // 计算中心点
    WCN_Point center = {0, 0};
    for (size_t i = 0; i < point_count; i++) {
      center.x += points[i].x;
      center.y += points[i].y;
    }
    center.x /= point_count;
    center.y /= point_count;

    // 创建三角形
    size_t max_vertices = (point_count - 1) * 3;
    WCN_Vertex *vertices =
        (WCN_Vertex *)malloc(max_vertices * sizeof(WCN_Vertex));
    size_t vertex_count = 0;

    for (size_t i = 0; i < point_count - 1 && vertex_count + 3 <= max_vertices;
         i++) {
      // 应用变换矩阵
      float cx = canvas->current_transform.m[0 * 4 + 0] * center.x +
                 canvas->current_transform.m[0 * 4 + 1] * center.y +
                 canvas->current_transform.m[0 * 4 + 2];
      float cy = canvas->current_transform.m[1 * 4 + 0] * center.x +
                 canvas->current_transform.m[1 * 4 + 1] * center.y +
                 canvas->current_transform.m[1 * 4 + 2];

      float p1x = canvas->current_transform.m[0 * 4 + 0] * points[i].x +
                  canvas->current_transform.m[0 * 4 + 1] * points[i].y +
                  canvas->current_transform.m[0 * 4 + 2];
      float p1y = canvas->current_transform.m[1 * 4 + 0] * points[i].x +
                  canvas->current_transform.m[1 * 4 + 1] * points[i].y +
                  canvas->current_transform.m[1 * 4 + 2];

      float p2x = canvas->current_transform.m[0 * 4 + 0] * points[i + 1].x +
                  canvas->current_transform.m[0 * 4 + 1] * points[i + 1].y +
                  canvas->current_transform.m[0 * 4 + 2];
      float p2y = canvas->current_transform.m[1 * 4 + 0] * points[i + 1].x +
                  canvas->current_transform.m[1 * 4 + 1] * points[i + 1].y +
                  canvas->current_transform.m[1 * 4 + 2];

      vertices[vertex_count++] =
          (WCN_Vertex){{cx, cy},
                       {canvas->fill_color.r, canvas->fill_color.g,
                        canvas->fill_color.b, canvas->fill_color.a}};
      vertices[vertex_count++] =
          (WCN_Vertex){{p1x, p1y},
                       {canvas->fill_color.r, canvas->fill_color.g,
                        canvas->fill_color.b, canvas->fill_color.a}};
      vertices[vertex_count++] =
          (WCN_Vertex){{p2x, p2y},
                       {canvas->fill_color.r, canvas->fill_color.g,
                        canvas->fill_color.b, canvas->fill_color.a}};
    }

    draw_vertices(canvas, vertices, vertex_count);
    free(vertices);
  }

  free(points);
}

// 描边路径
void wcn_stroke_path(WCN_Canvas *canvas) {
  if (!canvas || !canvas->pass || canvas->path_ops_count == 0) {
    return;
  }

  // 将路径转换为点序列
  size_t max_points = canvas->path_ops_count * 32; // 为曲线预留更多点
  WCN_Point *points = (WCN_Point *)malloc(max_points * sizeof(WCN_Point));
  size_t point_count = 0;

  WCN_Point current_pos = {0, 0};
  WCN_Point path_start = {0, 0};
  bool path_started = false;

  for (size_t i = 0; i < canvas->path_ops_count && point_count < max_points;
       i++) {
    switch (canvas->path_ops[i].type) {
    case WCN_PATH_MOVE_TO:
      current_pos = canvas->path_ops[i].data.point;
      path_start = current_pos;
      path_started = true;
      if (point_count < max_points) {
        points[point_count++] = current_pos;
      }
      break;

    case WCN_PATH_LINE_TO:
      current_pos = canvas->path_ops[i].data.point;
      if (point_count < max_points) {
        points[point_count++] = current_pos;
      }
      break;

    case WCN_PATH_BEZIER_CURVE_TO: {
      WCN_Point p0 = current_pos;
      WCN_Point p1 = canvas->path_ops[i].data.bezier.cp1;
      WCN_Point p2 = canvas->path_ops[i].data.bezier.cp2;
      WCN_Point p3 = canvas->path_ops[i].data.bezier.end;

      // 将贝塞尔曲线分割为多个线段
      int segments = 16;
      for (int j = 1; j <= segments && point_count < max_points; j++) {
        float t = (float)j / segments;
        points[point_count++] = bezier_point(p0, p1, p2, p3, t);
      }
      current_pos = p3;
      break;
    }

    case WCN_PATH_QUADRATIC_CURVE_TO: {
      WCN_Point p0 = current_pos;
      WCN_Point p1 = canvas->path_ops[i].data.quadratic.cp;
      WCN_Point p2 = canvas->path_ops[i].data.quadratic.end;

      // 将二次贝塞尔曲线分割为多个线段
      int segments = 16;
      for (int j = 1; j <= segments && point_count < max_points; j++) {
        float t = (float)j / segments;
        points[point_count++] = quadratic_bezier_point(p0, p1, p2, t);
      }
      current_pos = p2;
      break;
    }

    case WCN_PATH_ARC: {
      WCN_Point center = canvas->path_ops[i].data.arc.center;
      float radius = canvas->path_ops[i].data.arc.radius;
      float start_angle = canvas->path_ops[i].data.arc.startAngle;
      float end_angle = canvas->path_ops[i].data.arc.endAngle;
      bool anticlockwise = canvas->path_ops[i].data.arc.anticlockwise;

      // 计算角度差
      float angle_diff = end_angle - start_angle;
      if (anticlockwise && angle_diff > 0) {
        angle_diff -= 2 * M_PI;
      } else if (!anticlockwise && angle_diff < 0) {
        angle_diff += 2 * M_PI;
      }

      // 将弧线分割为多个线段
      int segments = (int)(fabsf(angle_diff) * 16 / M_PI);
      if (segments < 4)
        segments = 4;
      if (segments > 64)
        segments = 64;

      for (int j = 1; j <= segments && point_count < max_points; j++) {
        float t = (float)j / segments;
        float angle = start_angle + t * angle_diff;
        points[point_count++] = arc_point(center, radius, angle);
      }
      current_pos = arc_point(center, radius, end_angle);
      break;
    }

    case WCN_PATH_CLOSE:
      if (path_started && point_count < max_points) {
        points[point_count++] = path_start;
      }
      break;
    }
  }

  // 创建线条段（使用三角形化的线段）
  if (point_count >= 2) {
    size_t max_vertices =
        (point_count - 1) * 6; // 每条线段需要6个顶点（2个三角形）
    WCN_Vertex *vertices =
        (WCN_Vertex *)malloc(max_vertices * sizeof(WCN_Vertex));
    size_t vertex_count = 0;

    for (size_t i = 0; i < point_count - 1 && vertex_count + 6 <= max_vertices;
         i++) {
      // 应用变换矩阵
      float p1x = canvas->current_transform.m[0 * 4 + 0] * points[i].x +
                  canvas->current_transform.m[0 * 4 + 1] * points[i].y +
                  canvas->current_transform.m[0 * 4 + 2];
      float p1y = canvas->current_transform.m[1 * 4 + 0] * points[i].x +
                  canvas->current_transform.m[1 * 4 + 1] * points[i].y +
                  canvas->current_transform.m[1 * 4 + 2];

      float p2x = canvas->current_transform.m[0 * 4 + 0] * points[i + 1].x +
                  canvas->current_transform.m[0 * 4 + 1] * points[i + 1].y +
                  canvas->current_transform.m[0 * 4 + 2];
      float p2y = canvas->current_transform.m[1 * 4 + 0] * points[i + 1].x +
                  canvas->current_transform.m[1 * 4 + 1] * points[i + 1].y +
                  canvas->current_transform.m[1 * 4 + 2];

      // 使用辅助函数创建三角形化的线段
      create_line_triangles(&vertices[vertex_count], p1x, p1y, p2x, p2y,
                            canvas->line_width, canvas->stroke_color);
      vertex_count += 6;
    }

    draw_vertices(canvas, vertices, vertex_count);
    free(vertices);
  }

  free(points);
}

// 裁剪路径
void wcn_clip_path(WCN_Canvas *canvas) {
  if (!canvas || !canvas->pass) {
    return;
  }

  // 设置裁剪区域
  // 在 WebGPU 中，裁剪是通过设置视口和剪刀矩形来实现的
  // 基于当前路径计算边界框并设置剪刀矩形
  if (canvas->path_ops_count > 0) {
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    for (size_t i = 0; i < canvas->path_ops_count; i++) {
      WCN_Point point = {0, 0};
      bool point_set = false;

      switch (canvas->path_ops[i].type) {
      case WCN_PATH_MOVE_TO:
      case WCN_PATH_LINE_TO:
        point = canvas->path_ops[i].data.point;
        point_set = true;
        break;
      default:
        break;
      }

      if (point_set) {
        if (point.x < min_x)
          min_x = point.x;
        if (point.y < min_y)
          min_y = point.y;
        if (point.x > max_x)
          max_x = point.x;
        if (point.y > max_y)
          max_y = point.y;
      }
    }

    // 确保坐标在有效范围内
    if (min_x < max_x && min_y < max_y) {
      min_x = fmaxf(0, min_x);
      min_y = fmaxf(0, min_y);
      max_x = fminf(canvas->width, max_x);
      max_y = fminf(canvas->height, max_y);

      // 设置剪刀矩形
      wgpuRenderPassEncoderSetScissorRect(
          canvas->pass, (uint32_t)min_x, (uint32_t)min_y,
          (uint32_t)(max_x - min_x), (uint32_t)(max_y - min_y));
    }
  }
}

// 平移坐标系
void wcn_translate(WCN_Canvas *canvas, float x, float y) {
  if (!canvas || !canvas->pass) {
    return;
  }

  // 使用 WMATH_TRANSLATION(Mat3)() 创建平移矩阵
  WMATH_TYPE(Mat3) translation = WMATH_TRANSLATION(Mat3)(
      INIT$(Vec2, .v_x = x, .v_y = y)
  );

  // 使用 WMATH_MULTIPLY(Mat3)() 进行矩阵乘法
  canvas->current_transform = WMATH_MULTIPLY(Mat3)(
      canvas->current_transform, 
      translation
  );
}

// 旋转坐标系
void wcn_rotate(WCN_Canvas *canvas, float angle) {
  if (!canvas || !canvas->pass) {
    return;
  }

  // 使用 WMATH_ROTATION(Mat3)() 创建旋转矩阵
  WMATH_TYPE(Mat3) rotation = WMATH_ROTATION(Mat3)(angle);

  // 使用 WMATH_MULTIPLY(Mat3)() 进行矩阵乘法
  canvas->current_transform = WMATH_MULTIPLY(Mat3)(
      canvas->current_transform, 
      rotation
  );
}

// 缩放坐标系
void wcn_scale(WCN_Canvas *canvas, float x, float y) {
  if (!canvas || !canvas->pass) {
    return;
  }

  // 手动创建缩放矩阵
  WMATH_TYPE(Mat3) scale = WMATH_IDENTITY(Mat3)();
  scale.m[0] = x;  // m[0][0] - 缩放 x
  scale.m[5] = y;  // m[1][1] - 缩放 y

  // 使用 WMATH_MULTIPLY(Mat3)() 进行矩阵乘法
  canvas->current_transform = WMATH_MULTIPLY(Mat3)(
      canvas->current_transform, 
      scale
  );
}

// 获取 WebGPU 渲染通道描述符
WGPURenderPassDescriptor wcn_get_render_pass_descriptor(WCN_Canvas *canvas) {
  WGPURenderPassDescriptor renderPassDesc = {0};

  if (canvas && canvas->view) {
    WGPURenderPassColorAttachment colorAttachment = {0};
    colorAttachment.view = canvas->view;
    colorAttachment.loadOp = WGPULoadOp_Clear;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0};

    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;
  }

  return renderPassDesc;
}

// 清除矩形区域
void wcn_clear_rect(WCN_Canvas *canvas, float x, float y, float width,
                    float height) {
  if (!canvas || !canvas->pass) {
    return;
  }

  // 保存当前填充颜色
  WCN_Color original_color = canvas->fill_color;

  // 设置透明颜色进行清除
  WCN_Color clear_color = {0.0f, 0.0f, 0.0f, 0.0f};
  wcn_set_fill_color(canvas, clear_color);

  // 填充矩形区域
  wcn_fill_rect(canvas, x, y, width, height);

  // 恢复原始填充颜色
  wcn_set_fill_color(canvas, original_color);
}

// 填充整个 Canvas
void wcn_fill(WCN_Canvas *canvas) {
  if (!canvas) {
    return;
  }

  wcn_fill_rect(canvas, 0, 0, canvas->width, canvas->height);
}

// 描边整个 Canvas
void wcn_stroke(WCN_Canvas *canvas) {
  if (!canvas) {
    return;
  }

  wcn_stroke_rect(canvas, 0, 0, canvas->width, canvas->height);
}

// 清除 Canvas
void wcn_clear(WCN_Canvas *canvas, WCN_Color color) {
  if (!canvas) {
    return;
  }

  // 保存当前填充颜色
  WCN_Color original_color = canvas->fill_color;

  // 设置清除颜色
  wcn_set_fill_color(canvas, color);

  // 填充整个画布
  wcn_fill(canvas);

  // 立即刷新批渲染，确保清除操作立即执行
  wcn_flush_batch(canvas);

  // 恢复原始填充颜色
  wcn_set_fill_color(canvas, original_color);
}

// 开始路径
void wcn_begin_path(WCN_Canvas *canvas) {
  if (!canvas) {
    return;
  }

  // 重置路径操作数组
  canvas->path_ops_count = 0;
}

// 关闭路径
void wcn_close_path(WCN_Canvas *canvas) {
  if (!canvas) {
    return;
  }

  // 添加关闭路径操作
  ensure_path_ops_capacity(canvas, canvas->path_ops_count + 1);

  WCN_PathOp close_op;
  close_op.type = WCN_PATH_CLOSE;

  canvas->path_ops[canvas->path_ops_count] = close_op;
  canvas->path_ops_count++;
}

// 移动到指定点
void wcn_move_to(WCN_Canvas *canvas, float x, float y) {
  if (!canvas) {
    return;
  }

  // 添加移动操作
  ensure_path_ops_capacity(canvas, canvas->path_ops_count + 1);

  WCN_PathOp move_op;
  move_op.type = WCN_PATH_MOVE_TO;
  move_op.data.point.x = x;
  move_op.data.point.y = y;

  canvas->path_ops[canvas->path_ops_count] = move_op;
  canvas->path_ops_count++;
}

// 画线到指定点
void wcn_line_to(WCN_Canvas *canvas, float x, float y) {
  if (!canvas) {
    return;
  }

  // 添加线条操作
  ensure_path_ops_capacity(canvas, canvas->path_ops_count + 1);

  WCN_PathOp line_op;
  line_op.type = WCN_PATH_LINE_TO;
  line_op.data.point.x = x;
  line_op.data.point.y = y;

  canvas->path_ops[canvas->path_ops_count] = line_op;
  canvas->path_ops_count++;
}

// 画贝塞尔曲线
void wcn_bezier_curve_to(WCN_Canvas *canvas, float cp1x, float cp1y, float cp2x,
                         float cp2y, float x, float y) {
  if (!canvas) {
    return;
  }

  // 添加贝塞尔曲线操作
  ensure_path_ops_capacity(canvas, canvas->path_ops_count + 1);

  WCN_PathOp bezier_op;
  bezier_op.type = WCN_PATH_BEZIER_CURVE_TO;
  bezier_op.data.bezier.cp1.x = cp1x;
  bezier_op.data.bezier.cp1.y = cp1y;
  bezier_op.data.bezier.cp2.x = cp2x;
  bezier_op.data.bezier.cp2.y = cp2y;
  bezier_op.data.bezier.end.x = x;
  bezier_op.data.bezier.end.y = y;

  canvas->path_ops[canvas->path_ops_count] = bezier_op;
  canvas->path_ops_count++;
}

// 画二次贝塞尔曲线
void wcn_quadratic_curve_to(WCN_Canvas *canvas, float cpx, float cpy, float x,
                            float y) {
  if (!canvas) {
    return;
  }

  // 添加二次贝塞尔曲线操作
  ensure_path_ops_capacity(canvas, canvas->path_ops_count + 1);

  WCN_PathOp quadratic_op;
  quadratic_op.type = WCN_PATH_QUADRATIC_CURVE_TO;
  quadratic_op.data.quadratic.cp.x = cpx;
  quadratic_op.data.quadratic.cp.y = cpy;
  quadratic_op.data.quadratic.end.x = x;
  quadratic_op.data.quadratic.end.y = y;

  canvas->path_ops[canvas->path_ops_count] = quadratic_op;
  canvas->path_ops_count++;
}

// 画弧线
void wcn_arc(WCN_Canvas *canvas, float x, float y, float radius,
             float startAngle, float endAngle, bool anticlockwise) {
  if (!canvas) {
    return;
  }

  // 添加弧线操作
  ensure_path_ops_capacity(canvas, canvas->path_ops_count + 1);

  WCN_PathOp arc_op;
  arc_op.type = WCN_PATH_ARC;
  arc_op.data.arc.center.x = x;
  arc_op.data.arc.center.y = y;
  arc_op.data.arc.radius = radius;
  arc_op.data.arc.startAngle = startAngle;
  arc_op.data.arc.endAngle = endAngle;
  arc_op.data.arc.anticlockwise = anticlockwise;

  canvas->path_ops[canvas->path_ops_count] = arc_op;
  canvas->path_ops_count++;
}

// 提交渲染命令
void wcn_submit(WCN_Canvas *canvas) {
  if (!canvas || !canvas->encoder) {
    return;
  }

  // 结束任何未结束的渲染通道
  if (canvas->pass) {
    wgpuRenderPassEncoderEnd(canvas->pass);
    wgpuRenderPassEncoderRelease(canvas->pass);
    canvas->pass = NULL;
  }

  // 提交命令
  WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(canvas->encoder, NULL);
  if (cmdBuffer) {
    WGPUQueue queue = wgpuDeviceGetQueue(canvas->device);
    wgpuQueueSubmit(queue, 1, &cmdBuffer);
    wgpuCommandBufferRelease(cmdBuffer);
  }

  // 释放命令编码器
  wgpuCommandEncoderRelease(canvas->encoder);
  canvas->encoder = NULL;
}