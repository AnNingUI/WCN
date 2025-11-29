#ifndef WCN_STATE_MANAGER_WGSL_H
#define WCN_STATE_MANAGER_WGSL_H

#include "WCN/WCN_WGSL.h"

// 状态管理计算着色器
static const char* WCN_STATE_MANAGER_WGSL = WGSL_CODE(
// GPU状态结构体
struct GPUState {
    transform_matrix: mat4x4<f32>,
    fill_color: u32,
    stroke_color: u32,
    stroke_width: f32,
    global_alpha: f32,
    blend_mode: u32,
    state_flags: u32,
    reserved: array<f32, 4>,
};

);

#endif // WCN_STATE_MANAGER_WGSL_H