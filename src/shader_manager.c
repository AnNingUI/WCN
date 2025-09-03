#include "shader_manager.h"
#include "WCN/WCN.h"  // 包含WCN头文件以使用调试宏
#include <stdlib.h>
#include <string.h>

// 顶点着色器代码 (WGSL)
static const char *vertex_shader_wgsl =
    "struct VertexInput {\n"
    "    @location(0) position: vec2<f32>,\n"
    "    @location(1) color: vec4<f32>,\n"
    "}\n"
    "\n"
    "struct VertexOutput {\n"
    "    @builtin(position) position: vec4<f32>,\n"
    "    @location(0) color: vec4<f32>,\n"
    "}\n"
    "\n"
    "struct Uniforms {\n"
    "    resolution: vec2<f32>,\n"
    "}\n"
    "\n"
    "@group(0) @binding(0) var<uniform> uniforms: Uniforms;\n"
    "\n"
    "@vertex\n"
    "fn main(input: VertexInput) -> VertexOutput {\n"
    "    var output: VertexOutput;\n"
    "    // 将像素坐标转换为裁剪空间坐标\n"
    "    let x = (input.position.x / uniforms.resolution.x) * 2.0 - 1.0;\n"
    "    let y = 1.0 - (input.position.y / uniforms.resolution.y) * 2.0;\n"
    "    // 确保z和w分量正确设置\n"
    "    output.position = vec4<f32>(x, y, 0.0, 1.0);\n"
    "    output.color = input.color;\n"
    "    return output;\n"
    "}\n";

// 片段着色器代码 (WGSL)
static const char *fragment_shader_wgsl =
    "struct FragmentInput {\n"
    "    @location(0) color: vec4<f32>,\n"
    "}\n"
    "\n"
    "@fragment\n"
    "fn main(input: FragmentInput) -> @location(0) vec4<f32> {\n"
    "    return input.color;\n"
    "}\n";

// 初始化着色器管理器
WCN_ShaderManager *wcn_shader_manager_init(WGPUDevice device) {
  if (!device) {
    return NULL;
  }

  WCN_ShaderManager *manager =
      (WCN_ShaderManager *)malloc(sizeof(WCN_ShaderManager));
  if (!manager) {
    return NULL;
  }

  memset(manager, 0, sizeof(WCN_ShaderManager));
  manager->device = device;

  // 创建顶点着色器
  WGPUShaderModuleDescriptor vertex_desc = {0};
  // 正确设置标签
  WGPUStringView vertex_label = {.data = "Vertex Shader", .length = 13};
  vertex_desc.label = vertex_label;

  // 使用 WGSL 代码创建着色器
  WGPUShaderSourceWGSL wgsl_source = {0};
  wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
  WGPUStringView code_view = {.data = vertex_shader_wgsl,
                              .length = strlen(vertex_shader_wgsl)};
  wgsl_source.code = code_view;

  vertex_desc.nextInChain = (WGPUChainedStruct *)&wgsl_source;
  manager->color_shader.vertex_shader =
      wgpuDeviceCreateShaderModule(device, &vertex_desc);

  // 检查顶点着色器是否创建成功
  if (!manager->color_shader.vertex_shader) {
    WCN_DEBUG_PRINT("Failed to create vertex shader module");
    free(manager);
    return NULL;
  }

  // 创建片段着色器
  WGPUShaderModuleDescriptor fragment_desc = {0};
  // 正确设置标签
  WGPUStringView fragment_label = {.data = "Fragment Shader", .length = 15};
  fragment_desc.label = fragment_label;

  // 使用 WGSL 代码创建着色器
  WGPUShaderSourceWGSL fragment_wgsl_source = {0};
  fragment_wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
  WGPUStringView fragment_code_view = {.data = fragment_shader_wgsl,
                                       .length = strlen(fragment_shader_wgsl)};
  fragment_wgsl_source.code = fragment_code_view;

  fragment_desc.nextInChain = (WGPUChainedStruct *)&fragment_wgsl_source;
  manager->color_shader.fragment_shader =
      wgpuDeviceCreateShaderModule(device, &fragment_desc);

  // 检查片段着色器是否创建成功
  if (!manager->color_shader.fragment_shader) {
    WCN_DEBUG_PRINT("Failed to create fragment shader module");
    if (manager->color_shader.vertex_shader) {
      wgpuShaderModuleRelease(manager->color_shader.vertex_shader);
    }
    free(manager);
    return NULL;
  }

  return manager;
}

// 销毁着色器管理器
void wcn_shader_manager_destroy(WCN_ShaderManager *manager) {
  if (manager) {
    if (manager->color_shader.vertex_shader) {
      wgpuShaderModuleRelease(manager->color_shader.vertex_shader);
    }
    if (manager->color_shader.fragment_shader) {
      wgpuShaderModuleRelease(manager->color_shader.fragment_shader);
    }
    free(manager);
  }
}

// 获取颜色着色器
WCN_ShaderModule
wcn_shader_manager_get_color_shader(WCN_ShaderManager *manager) {
  if (manager) {
    return manager->color_shader;
  }

  WCN_ShaderModule empty = {0};
  return empty;
}