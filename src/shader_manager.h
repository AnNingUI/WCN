#ifndef SHADER_MANAGER_H
#define SHADER_MANAGER_H

#include <webgpu/webgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

// 着色器模块
typedef struct {
  WGPUShaderModule vertex_shader;
  WGPUShaderModule fragment_shader;
} WCN_ShaderModule;

// 着色器管理器
typedef struct {
  WGPUDevice device;
  WCN_ShaderModule color_shader;
  WCN_ShaderModule texture_shader;
} WCN_ShaderManager;

// 初始化着色器管理器
WCN_ShaderManager *wcn_shader_manager_init(WGPUDevice device);

// 销毁着色器管理器
void wcn_shader_manager_destroy(WCN_ShaderManager *manager);

// 获取颜色着色器
WCN_ShaderModule
wcn_shader_manager_get_color_shader(WCN_ShaderManager *manager);

#ifdef __cplusplus
}
#endif

#endif // SHADER_MANAGER_H