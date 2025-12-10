#include "wcn_math_internal.h"
#include "WCN/WCN_SIMD_MACROS.h"
// FROM

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_mat4)(const WMATH_TYPE(Mat4) a) {
  return WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
      .m_00 = a.m[0],
      .m_01 = a.m[1],
      .m_02 = a.m[2],
      .m_10 = a.m[4],
      .m_11 = a.m[5],
      .m_12 = a.m[6],
      .m_20 = a.m[8],
      .m_21 = a.m[9],
      .m_22 = a.m[10],
  });
}

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_quat)(WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // Load q = [x, y, z, w]
  __m128 q_vec = _mm_loadu_ps(q.v);
  
  // q2 = q + q = [2x, 2y, 2z, 2w]
  __m128 q2 = _mm_add_ps(q_vec, q_vec);
  
  // 计算所有积的组合: q_vec * q2
  // 我们需要: 2xx, 2yy, 2zz
  // 以及交叉项: 2xy, 2xz, 2yz, 2wx, 2wy, 2wz
  
  // Broadcast components
  __m128 q_x = _mm_shuffle_ps(q_vec, q_vec, 0x00);
  __m128 q_y = _mm_shuffle_ps(q_vec, q_vec, 0x55);
  __m128 q_z = _mm_shuffle_ps(q_vec, q_vec, 0xAA);
  
  // Products
  __m128 qx_q2 = _mm_mul_ps(q_x, q2); // [2xx, 2xy, 2xz, 2xw]
  __m128 qy_q2 = _mm_mul_ps(q_y, q2); // [2yx, 2yy, 2yz, 2yw]
  __m128 qz_q2 = _mm_mul_ps(q_z, q2); // [2zx, 2zy, 2zz, 2zw]
  
  // 准备构建列
  __m128 one = _mm_set1_ps(1.0f);
  
  // Col 0: [1-2yy-2zz, 2xy+2wz, 2xz-2wy]
  // yy + zz
  __m128 yy = _mm_shuffle_ps(qy_q2, qy_q2, 0x55);
  __m128 zz = _mm_shuffle_ps(qz_q2, qz_q2, 0xAA);
  __m128 diag0 = _mm_sub_ps(one, _mm_add_ps(yy, zz));
  
  // xy + wz (2xy is qx_q2[1], 2wz is qz_q2[3])
  __m128 xy = _mm_shuffle_ps(qx_q2, qx_q2, 0x55);
  __m128 wz = _mm_shuffle_ps(qz_q2, qz_q2, 0xFF);
  __m128 m10 = _mm_add_ps(xy, wz);
  
  // xz - wy (2xz is qx_q2[2], 2wy is qy_q2[3])
  __m128 xz = _mm_shuffle_ps(qx_q2, qx_q2, 0xAA);
  __m128 wy = _mm_shuffle_ps(qy_q2, qy_q2, 0xFF);
  __m128 m20 = _mm_sub_ps(xz, wy);
  
  // Col 1: [2xy-2wz, 1-2xx-2zz, 2yz+2wx]
  // xy - wz
  __m128 m01 = _mm_sub_ps(xy, wz);
  
  // 1 - xx - zz
  __m128 xx = _mm_shuffle_ps(qx_q2, qx_q2, 0x00);
  __m128 diag1 = _mm_sub_ps(one, _mm_add_ps(xx, zz));
  
  // yz + wx (2yz is qy_q2[2], 2wx is qx_q2[3])
  __m128 yz = _mm_shuffle_ps(qy_q2, qy_q2, 0xAA);
  __m128 wx = _mm_shuffle_ps(qx_q2, qx_q2, 0xFF);
  __m128 m21 = _mm_add_ps(yz, wx);
  
  // Col 2: [2xz+2wy, 2yz-2wx, 1-2xx-2yy]
  // xz + wy
  __m128 m02 = _mm_add_ps(xz, wy);
  
  // yz - wx
  __m128 m12 = _mm_sub_ps(yz, wx);
  
  // 1 - xx - yy
  __m128 diag2 = _mm_sub_ps(one, _mm_add_ps(xx, yy));
  
  // Pack and Store
  // Row packing is tricky for Mat3 (stride 4), easier to pack columns and store
  // m[0..3] = [diag0, m10, m20, 0]
  __m128 col0 = _mm_unpacklo_ps(diag0, m10); // [d0, m10, d0, m10]
  __m128 col0_hi = _mm_movelh_ps(m20, _mm_setzero_ps()); // [m20, 0, ...]
  col0 = _mm_shuffle_ps(col0, col0_hi, _MM_SHUFFLE(2, 0, 1, 0)); // [d0, m10, m20, 0]
  
  // m[4..7] = [m01, diag1, m21, 0]
  __m128 col1 = _mm_unpacklo_ps(m01, diag1);
  __m128 col1_hi = _mm_movelh_ps(m21, _mm_setzero_ps());
  col1 = _mm_shuffle_ps(col1, col1_hi, _MM_SHUFFLE(2, 0, 1, 0));
  
  // m[8..11] = [m02, m12, diag2, 0]
  __m128 col2 = _mm_unpacklo_ps(m02, m12);
  __m128 col2_hi = _mm_movelh_ps(diag2, _mm_setzero_ps());
  col2 = _mm_shuffle_ps(col2, col2_hi, _MM_SHUFFLE(2, 0, 1, 0));
  
  _mm_storeu_ps(&newDst.m[0], col0);
  _mm_storeu_ps(&newDst.m[4], col1);
  _mm_storeu_ps(&newDst.m[8], col2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // 1. Load & Prep
  float32x4_t q_vec = vld1q_f32(q.v);        // [x, y, z, w]
  float32x4_t q2    = vaddq_f32(q_vec, q_vec); // [2x, 2y, 2z, 2w]
  float32x4_t one   = vdupq_n_f32(1.0f);
  float32x4_t zero  = vdupq_n_f32(0.0f);

  // 2. Parallel Math (Implicit Broadcast via Lane Multiply)
  // ARM64 can multiply a vector by a scalar element of another vector directly
  float32x4_t qx_q2 = vmulq_laneq_f32(q2, q_vec, 0); // [2xx, 2xy, 2xz, 2xw]
  float32x4_t qy_q2 = vmulq_laneq_f32(q2, q_vec, 1); // [2yx, 2yy, 2yz, 2yw]
  float32x4_t qz_q2 = vmulq_laneq_f32(q2, q_vec, 2); // [2zx, 2zy, 2zz, 2zw]

  // 3. Construct Components (Splatting needed results)
  // Col 0
  float32x4_t yy = vdupq_laneq_f32(qy_q2, 1);
  float32x4_t zz = vdupq_laneq_f32(qz_q2, 2);
  float32x4_t diag0 = vsubq_f32(one, vaddq_f32(yy, zz));

  float32x4_t xy = vdupq_laneq_f32(qx_q2, 1);
  float32x4_t wz = vdupq_laneq_f32(qz_q2, 3);
  float32x4_t m10 = vaddq_f32(xy, wz);

  float32x4_t xz = vdupq_laneq_f32(qx_q2, 2);
  float32x4_t wy = vdupq_laneq_f32(qy_q2, 3);
  float32x4_t m20 = vsubq_f32(xz, wy);

  // Col 1
  float32x4_t m01 = vsubq_f32(xy, wz); // reused xy, wz

  float32x4_t xx = vdupq_laneq_f32(qx_q2, 0);
  float32x4_t diag1 = vsubq_f32(one, vaddq_f32(xx, zz));

  float32x4_t yz = vdupq_laneq_f32(qy_q2, 2);
  float32x4_t wx = vdupq_laneq_f32(qx_q2, 3);
  float32x4_t m21 = vaddq_f32(yz, wx);

  // Col 2
  float32x4_t m02 = vaddq_f32(xz, wy); // reused xz, wy
  float32x4_t m12 = vsubq_f32(yz, wx); // reused yz, wx
  float32x4_t diag2 = vsubq_f32(one, vaddq_f32(xx, yy));

  // 4. Pack & Store
  // Interleave to build columns: [v0, v1, v0, v1]
  // Col 0: [diag0, m10, m20, 0]
  float32x4_t c0_lo = vzip1q_f32(diag0, m10); // [d0, m10, d0, m10]
  float32x4_t c0_hi = vzip1q_f32(m20, zero);  // [m20, 0, m20, 0]
  // Combine low halves: [d0, m10] + [m20, 0]
  vst1q_f32(&newDst.m[0], vcombine_f32(vget_low_f32(c0_lo), vget_low_f32(c0_hi)));

  // Col 1: [m01, diag1, m21, 0]
  float32x4_t c1_lo = vzip1q_f32(m01, diag1);
  float32x4_t c1_hi = vzip1q_f32(m21, zero);
  vst1q_f32(&newDst.m[4], vcombine_f32(vget_low_f32(c1_lo), vget_low_f32(c1_hi)));

  // Col 2: [m02, m12, diag2, 0]
  float32x4_t c2_lo = vzip1q_f32(m02, m12);
  float32x4_t c2_hi = vzip1q_f32(diag2, zero);
  vst1q_f32(&newDst.m[8], vcombine_f32(vget_low_f32(c2_lo), vget_low_f32(c2_hi)));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // 1. Load & Prep
  v128_t q_vec = wasm_v128_load(q.v);
  v128_t q2    = wasm_f32x4_add(q_vec, q_vec);
  v128_t one   = wasm_f32x4_splat(1.0f);
  v128_t zero  = wasm_f32x4_splat(0.0f);

  // 2. Broadcast (Shuffle inputs)
  v128_t q_x = wasm_v32x4_shuffle(q_vec, q_vec, 0, 0, 0, 0);
  v128_t q_y = wasm_v32x4_shuffle(q_vec, q_vec, 1, 1, 1, 1);
  v128_t q_z = wasm_v32x4_shuffle(q_vec, q_vec, 2, 2, 2, 2);

  // 3. Products
  v128_t qx_q2 = wasm_f32x4_mul(q_x, q2); // [2xx, 2xy, 2xz, 2xw]
  v128_t qy_q2 = wasm_f32x4_mul(q_y, q2); // [2yx, 2yy, 2yz, 2yw]
  v128_t qz_q2 = wasm_f32x4_mul(q_z, q2); // [2zx, 2zy, 2zz, 2zw]

  // 4. Construct Terms
  // Extracts here are VIRTUAL (shuffles), not CPU extracts
  // Col 0
  v128_t yy = wasm_v32x4_shuffle(qy_q2, qy_q2, 1, 1, 1, 1);
  v128_t zz = wasm_v32x4_shuffle(qz_q2, qz_q2, 2, 2, 2, 2);
  v128_t diag0 = wasm_f32x4_sub(one, wasm_f32x4_add(yy, zz));

  v128_t xy = wasm_v32x4_shuffle(qx_q2, qx_q2, 1, 1, 1, 1);
  v128_t wz = wasm_v32x4_shuffle(qz_q2, qz_q2, 3, 3, 3, 3);
  v128_t m10 = wasm_f32x4_add(xy, wz);

  v128_t xz = wasm_v32x4_shuffle(qx_q2, qx_q2, 2, 2, 2, 2);
  v128_t wy = wasm_v32x4_shuffle(qy_q2, qy_q2, 3, 3, 3, 3);
  v128_t m20 = wasm_f32x4_sub(xz, wy);

  // Col 1
  v128_t m01 = wasm_f32x4_sub(xy, wz);
  
  v128_t xx = wasm_v32x4_shuffle(qx_q2, qx_q2, 0, 0, 0, 0);
  v128_t diag1 = wasm_f32x4_sub(one, wasm_f32x4_add(xx, zz));

  v128_t yz = wasm_v32x4_shuffle(qy_q2, qy_q2, 2, 2, 2, 2);
  v128_t wx = wasm_v32x4_shuffle(qx_q2, qx_q2, 3, 3, 3, 3);
  v128_t m21 = wasm_f32x4_add(yz, wx);

  // Col 2
  v128_t m02 = wasm_f32x4_add(xz, wy);
  v128_t m12 = wasm_f32x4_sub(yz, wx);
  v128_t diag2 = wasm_f32x4_sub(one, wasm_f32x4_add(xx, yy));

  // 5. Pack Columns (Shuffle Mix)
  // Col 0: [d0, m10, m20, 0]
  // Shuffle logic: Takes indices 0-3 from arg1, 4-7 from arg2
  // We construct [d0, m10] from lo(diag0, m10), then merge with [m20, 0]
  // WASM doesn't have an easy "zip", must use specific indices.
  // Assuming inputs are splatted vectors (all lanes same):
  // We want lane 0 from diag0, lane 0 from m10, lane 0 from m20, lane 0 from zero
  
  // Create Col0: 0 from diag0, 4 from m10, 0 from m20, 4 from zero (re-ordered via nested shuffles)
  // Or simpler:
  v128_t col0_lo = wasm_v32x4_shuffle(diag0, m10, 0, 4, 0, 4); // [d0, m10, d0, m10]
  v128_t col0_hi = wasm_v32x4_shuffle(m20, zero, 0, 4, 0, 4);  // [m20, 0, m20, 0]
  v128_t col0 = wasm_v32x4_shuffle(col0_lo, col0_hi, 0, 1, 4, 5);
  wasm_v128_store(&newDst.m[0], col0);

  // Col 1
  v128_t col1_lo = wasm_v32x4_shuffle(m01, diag1, 0, 4, 0, 4);
  v128_t col1_hi = wasm_v32x4_shuffle(m21, zero, 0, 4, 0, 4);
  v128_t col1 = wasm_v32x4_shuffle(col1_lo, col1_hi, 0, 1, 4, 5);
  wasm_v128_store(&newDst.m[4], col1);

  // Col 2
  v128_t col2_lo = wasm_v32x4_shuffle(m02, m12, 0, 4, 0, 4);
  v128_t col2_hi = wasm_v32x4_shuffle(diag2, zero, 0, 4, 0, 4);
  v128_t col2 = wasm_v32x4_shuffle(col2_lo, col2_hi, 0, 1, 4, 5);
  wasm_v128_store(&newDst.m[8], col2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // Need to set vector length to 4 for 32-bit floats
  size_t vl = __riscv_vsetvl_e32m1(4);

  // 1. Load & Prep
  vfloat32m1_t q_vec = __riscv_vle32_v_f32m1(q.v, vl);
  vfloat32m1_t q2    = __riscv_vfadd_vv_f32m1(q_vec, q_vec, vl);
  vfloat32m1_t one   = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  vfloat32m1_t zero  = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  // 2. Broadcast (Splat x, y, z from q_vec)
  // Gather indices: [0,0,0,0], [1,1,1,1]...
  vuint32m1_t idx_0 = __riscv_vmv_v_x_u32m1(0, vl);
  vuint32m1_t idx_1 = __riscv_vmv_v_x_u32m1(1, vl);
  vuint32m1_t idx_2 = __riscv_vmv_v_x_u32m1(2, vl);
  vuint32m1_t idx_3 = __riscv_vmv_v_x_u32m1(3, vl);

  vfloat32m1_t q_x = __riscv_vrgather_vv_f32m1(q_vec, idx_0, vl);
  vfloat32m1_t q_y = __riscv_vrgather_vv_f32m1(q_vec, idx_1, vl);
  vfloat32m1_t q_z = __riscv_vrgather_vv_f32m1(q_vec, idx_2, vl);

  // 3. Products
  vfloat32m1_t qx_q2 = __riscv_vfmul_vv_f32m1(q_x, q2, vl);
  vfloat32m1_t qy_q2 = __riscv_vfmul_vv_f32m1(q_y, q2, vl);
  vfloat32m1_t qz_q2 = __riscv_vfmul_vv_f32m1(q_z, q2, vl);

  // 4. Construct Components (Extract via Gather)
  // Col 0
  vfloat32m1_t yy = __riscv_vrgather_vv_f32m1(qy_q2, idx_1, vl);
  vfloat32m1_t zz = __riscv_vrgather_vv_f32m1(qz_q2, idx_2, vl);
  vfloat32m1_t diag0 = __riscv_vfsub_vv_f32m1(one, __riscv_vfadd_vv_f32m1(yy, zz, vl), vl);

  vfloat32m1_t xy = __riscv_vrgather_vv_f32m1(qx_q2, idx_1, vl);
  vfloat32m1_t wz = __riscv_vrgather_vv_f32m1(qz_q2, idx_3, vl);
  vfloat32m1_t m10 = __riscv_vfadd_vv_f32m1(xy, wz, vl);

  vfloat32m1_t xz = __riscv_vrgather_vv_f32m1(qx_q2, idx_2, vl);
  vfloat32m1_t wy = __riscv_vrgather_vv_f32m1(qy_q2, idx_3, vl);
  vfloat32m1_t m20 = __riscv_vfsub_vv_f32m1(xz, wy, vl);

  // Col 1
  vfloat32m1_t m01 = __riscv_vfsub_vv_f32m1(xy, wz, vl);

  vfloat32m1_t xx = __riscv_vrgather_vv_f32m1(qx_q2, idx_0, vl);
  vfloat32m1_t diag1 = __riscv_vfsub_vv_f32m1(one, __riscv_vfadd_vv_f32m1(xx, zz, vl), vl);

  vfloat32m1_t yz = __riscv_vrgather_vv_f32m1(qy_q2, idx_2, vl);
  vfloat32m1_t wx = __riscv_vrgather_vv_f32m1(qx_q2, idx_3, vl);
  vfloat32m1_t m21 = __riscv_vfadd_vv_f32m1(yz, wx, vl);

  // Col 2
  vfloat32m1_t m02 = __riscv_vfadd_vv_f32m1(xz, wy, vl);
  vfloat32m1_t m12 = __riscv_vfsub_vv_f32m1(yz, wx, vl);
  vfloat32m1_t diag2 = __riscv_vfsub_vv_f32m1(one, __riscv_vfadd_vv_f32m1(xx, yy, vl), vl);

  // 5. Pack and Store
  // We have vectors where all lanes are same. We need to merge them.
  // Col 0 Target: [diag0[0], m10[0], m20[0], 0]
  // We can use vslideup / vmerge, or just vrgather again with a composite index vector?
  // Since we are writing to memory, if we can't easily zip, we can construct a "column vector"
  // using sliding.
  
  // Strategy: Start with zero, slide in components.
  // dest = zero
  // dest[0] = diag0[0] -> vmerge (masked) or vslideup
  // Actually, cleanest way in RVV without complex shuffles is usually vslide.
  
  // Col 0
  vfloat32m1_t c0 = zero; // lane 3 is 0
  c0 = __riscv_vslideup_vx_f32m1(c0, m20, 2, vl); // lane 2 = m20[0]
  c0 = __riscv_vslideup_vx_f32m1(c0, m10, 1, vl); // lane 1 = m10[0]
  c0 = __riscv_vslideup_vx_f32m1(c0, diag0, 0, vl); // lane 0 = diag0[0]
  __riscv_vse32_v_f32m1(&newDst.m[0], c0, vl);

  // Col 1
  vfloat32m1_t c1 = zero;
  c1 = __riscv_vslideup_vx_f32m1(c1, m21, 2, vl);
  c1 = __riscv_vslideup_vx_f32m1(c1, diag1, 1, vl);
  c1 = __riscv_vslideup_vx_f32m1(c1, m01, 0, vl);
  __riscv_vse32_v_f32m1(&newDst.m[4], c1, vl);

  // Col 2
  vfloat32m1_t c2 = zero;
  c2 = __riscv_vslideup_vx_f32m1(c2, diag2, 2, vl);
  c2 = __riscv_vslideup_vx_f32m1(c2, m12, 1, vl);
  c2 = __riscv_vslideup_vx_f32m1(c2, m02, 0, vl);
  __riscv_vse32_v_f32m1(&newDst.m[8], c2, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // 1. Load & Prep
  __m128 q_vec = __lsx_vld(q.v, 0);
  __m128 q2    = __lsx_vfadd_s(q_vec, q_vec);
  __m128 one   = __lsx_vldi(0x3F80); // Load 1.0 directly (if imm supported) or splat
  // If vldi is specific to integer, use repl:
  if (1) { float f=1.0f; one = __lsx_vldrepl_w(&f, 0); } // Fallback safe splat
  __m128 zero  = __lsx_vldi(0);

  // 2. Broadcast & Products
  // __lsx_vreplve_w: Replicate word from vector element
  __m128 q_x = __lsx_vreplve_w(q_vec, 0);
  __m128 q_y = __lsx_vreplve_w(q_vec, 1);
  __m128 q_z = __lsx_vreplve_w(q_vec, 2);

  __m128 qx_q2 = __lsx_vfmul_s(q_x, q2);
  __m128 qy_q2 = __lsx_vfmul_s(q_y, q2);
  __m128 qz_q2 = __lsx_vfmul_s(q_z, q2);

  // 3. Construct Components
  // Col 0
  __m128 yy = __lsx_vreplve_w(qy_q2, 1);
  __m128 zz = __lsx_vreplve_w(qz_q2, 2);
  __m128 diag0 = __lsx_vfsub_s(one, __lsx_vfadd_s(yy, zz));

  __m128 xy = __lsx_vreplve_w(qx_q2, 1);
  __m128 wz = __lsx_vreplve_w(qz_q2, 3);
  __m128 m10 = __lsx_vfadd_s(xy, wz);

  __m128 xz = __lsx_vreplve_w(qx_q2, 2);
  __m128 wy = __lsx_vreplve_w(qy_q2, 3);
  __m128 m20 = __lsx_vfsub_s(xz, wy);

  // Col 1
  __m128 m01 = __lsx_vfsub_s(xy, wz);

  __m128 xx = __lsx_vreplve_w(qx_q2, 0);
  __m128 diag1 = __lsx_vfsub_s(one, __lsx_vfadd_s(xx, zz));

  __m128 yz = __lsx_vreplve_w(qy_q2, 2);
  __m128 wx = __lsx_vreplve_w(qx_q2, 3);
  __m128 m21 = __lsx_vfadd_s(yz, wx);

  // Col 2
  __m128 m02 = __lsx_vfadd_s(xz, wy);
  __m128 m12 = __lsx_vfsub_s(yz, wx);
  __m128 diag2 = __lsx_vfsub_s(one, __lsx_vfadd_s(xx, yy)); // yy reused from Col0 logic

  // 4. Pack & Store
  // LSX has vilvl/vilvh (interleave low/high) similar to unpack
  // Col 0: [d0, m10, m20, 0]
  __m128 c0_lo = __lsx_vilvl_w(m10, diag0); // [d0, m10, d0, m10] (args reversed vs unpacklo usually)
  __m128 c0_hi = __lsx_vilvl_w(zero, m20);  // [m20, 0, m20, 0]
  __m128 col0  = __lsx_vilvl_d(c0_hi, c0_lo); // Combine 64-bit halves
  __lsx_vst(col0, &newDst.m[0], 0);

  // Col 1: [m01, diag1, m21, 0]
  __m128 c1_lo = __lsx_vilvl_w(diag1, m01);
  __m128 c1_hi = __lsx_vilvl_w(zero, m21);
  __m128 col1  = __lsx_vilvl_d(c1_hi, c1_lo);
  __lsx_vst(col1, &newDst.m[4], 0);

  // Col 2: [m02, m12, diag2, 0]
  __m128 c2_lo = __lsx_vilvl_w(m12, m02);
  __m128 c2_hi = __lsx_vilvl_w(zero, diag2);
  __m128 col2  = __lsx_vilvl_d(c2_hi, c2_lo);
  __lsx_vst(col2, &newDst.m[8], 0);

#else
  // Scalar fallback
  float x = q.v[0];
  float y = q.v[1];
  float z = q.v[2];
  float w = q.v[3];
  float x2 = x + x;
  float y2 = y + y;
  float z2 = z + z;

  float xx = x * x2;
  float yx = y * x2;
  float yy = y * y2;
  float zx = z * x2;
  float zy = z * y2;
  float zz = z * z2;
  float wx = w * x2;
  float wy = w * y2;
  float wz = w * z2;

  newDst.m[0] = 1 - yy - zz;
  newDst.m[1] = yx + wz;
  newDst.m[2] = zx - wy;
  newDst.m[3] = 0;
  newDst.m[4] = yx - wz;
  newDst.m[5] = 1 - xx - zz;
  newDst.m[6] = zy + wx;
  newDst.m[7] = 0;
  newDst.m[8] = zx + wy;
  newDst.m[9] = zy - wx;
  newDst.m[10] = 1 - xx - yy;
  newDst.m[11] = 0;
#endif

  return newDst;
}

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_mat3)(const WMATH_TYPE(Mat3) a) {
  WMATH_TYPE(Mat4) m;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // 1. Copy Col 0, 1, 2 directly (Assume input stride is 4 floats)
  // Note: This copies the w-component (padding) from Mat3. 
  // If Mat3 padding is dirty, use _mm_and_ps with a mask to zero it, but usually it's clean or ignored.
  _mm_storeu_ps(&m.m[0], _mm_loadu_ps(&a.m[0]));
  _mm_storeu_ps(&m.m[4], _mm_loadu_ps(&a.m[4]));
  _mm_storeu_ps(&m.m[8], _mm_loadu_ps(&a.m[8]));
  
  // 2. Set Col 3 to [0, 0, 0, 1]
  // _mm_set_ps inputs are in reverse order (w, z, y, x) -> [0, 0, 0, 1] in memory
  _mm_storeu_ps(&m.m[12], _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // 1. Copy first 3 columns
  vst1q_f32(&m.m[0], vld1q_f32(&a.m[0]));
  vst1q_f32(&m.m[4], vld1q_f32(&a.m[4]));
  vst1q_f32(&m.m[8], vld1q_f32(&a.m[8]));
  
  // 2. Set Col 3 to [0, 0, 0, 1]
  // Using static const array is often faster/safer than per-element construction
  static const float c3_data[] = {0.0f, 0.0f, 0.0f, 1.0f};
  vst1q_f32(&m.m[12], vld1q_f32(c3_data));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // 1. Copy first 3 columns
  wasm_v128_store(&m.m[0], wasm_v128_load(&a.m[0]));
  wasm_v128_store(&m.m[4], wasm_v128_load(&a.m[4]));
  wasm_v128_store(&m.m[8], wasm_v128_load(&a.m[8]));

  // 2. Set Col 3 to [0, 0, 0, 1]
  // wasm_f32x4_make args are (x, y, z, w)
  v128_t c3 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 1.0f);
  wasm_v128_store(&m.m[12], c3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // 1. Setup VL
  size_t vl = __riscv_vsetvl_e32m1(4);

  // 2. Copy first 3 columns
  vfloat32m1_t c0 = __riscv_vle32_v_f32m1(&a.m[0], vl);
  __riscv_vse32_v_f32m1(&m.m[0], c0, vl);

  vfloat32m1_t c1 = __riscv_vle32_v_f32m1(&a.m[4], vl);
  __riscv_vse32_v_f32m1(&m.m[4], c1, vl);

  vfloat32m1_t c2 = __riscv_vle32_v_f32m1(&a.m[8], vl);
  __riscv_vse32_v_f32m1(&m.m[8], c2, vl);

  // 3. Set Col 3
  // Loading from constant pool is efficient
  static const float c3_data[] = {0.0f, 0.0f, 0.0f, 1.0f};
  vfloat32m1_t c3 = __riscv_vle32_v_f32m1(c3_data, vl);
  __riscv_vse32_v_f32m1(&m.m[12], c3, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // 1. Copy first 3 columns
  __lsx_vst(__lsx_vld(&a.m[0], 0), &m.m[0], 0);
  __lsx_vst(__lsx_vld(&a.m[4], 0), &m.m[4], 0);
  __lsx_vst(__lsx_vld(&a.m[8], 0), &m.m[8], 0);

  // 2. Set Col 3
  static const float c3_data[] = {0.0f, 0.0f, 0.0f, 1.0f};
  __lsx_vst(__lsx_vld(c3_data, 0), &m.m[12], 0);

#else
  // Scalar Fallback
  // Explicitly zeroing the w components ensures correctness even if input padding is dirty
  m.m[ 0] = a.m[0]; m.m[ 1] = a.m[1]; m.m[ 2] = a.m[ 2];  m.m[ 3] = 0.0f;
  m.m[ 4] = a.m[4]; m.m[ 5] = a.m[5]; m.m[ 6] = a.m[ 6];  m.m[ 7] = 0.0f;
  m.m[ 8] = a.m[8]; m.m[ 9] = a.m[9]; m.m[10] = a.m[10];  m.m[11] = 0.0f;
  m.m[12] =   0.0f; m.m[13] =   0.0f; m.m[14] =    0.0f;  m.m[15] = 1.0f;
#endif

  return m;
}

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_quat)(const WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Mat4) m;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // Load q = [x, y, z, w]
  __m128 q_vec = _mm_loadu_ps(q.v);
  __m128 q2 = _mm_add_ps(q_vec, q_vec); // [2x, 2y, 2z, 2w]
  __m128 one = _mm_set1_ps(1.0f);
  
  // Broadcast components
  __m128 q_x = _mm_shuffle_ps(q_vec, q_vec, 0x00);
  __m128 q_y = _mm_shuffle_ps(q_vec, q_vec, 0x55);
  __m128 q_z = _mm_shuffle_ps(q_vec, q_vec, 0xAA);
  
  // Products [2xx, 2xy, 2xz, 2xw], etc.
  __m128 qx_q2 = _mm_mul_ps(q_x, q2); 
  __m128 qy_q2 = _mm_mul_ps(q_y, q2); 
  __m128 qz_q2 = _mm_mul_ps(q_z, q2); 
  
  // --- Row 0: [1-2yy-2zz, 2xy+2wz, 2xz-2wy, 0] ---
  __m128 yy = _mm_shuffle_ps(qy_q2, qy_q2, 0x55);
  __m128 zz = _mm_shuffle_ps(qz_q2, qz_q2, 0xAA);
  __m128 r0_diag = _mm_sub_ps(one, _mm_add_ps(yy, zz));
  
  __m128 xy = _mm_shuffle_ps(qx_q2, qx_q2, 0x55);
  __m128 wz = _mm_shuffle_ps(qz_q2, qz_q2, 0xFF);
  __m128 r0_x = _mm_add_ps(xy, wz); // yx + wz
  
  __m128 xz = _mm_shuffle_ps(qx_q2, qx_q2, 0xAA);
  __m128 wy = _mm_shuffle_ps(qy_q2, qy_q2, 0xFF);
  __m128 r0_y = _mm_sub_ps(xz, wy); // zx - wy
  
  // Pack Row 0: [r0_diag, r0_x, r0_y, 0]
  __m128 r0_tmp1 = _mm_unpacklo_ps(r0_diag, r0_x);
  __m128 r0_tmp2 = _mm_movelh_ps(r0_y, _mm_setzero_ps());
  __m128 row0 = _mm_shuffle_ps(r0_tmp1, r0_tmp2, _MM_SHUFFLE(2, 0, 1, 0));
  _mm_storeu_ps(&m.m[0], row0);
  
  // --- Row 1: [2xy-2wz, 1-2xx-2zz, 2yz+2wx, 0] ---
  __m128 r1_x = _mm_sub_ps(xy, wz); // yx - wz
  
  __m128 xx = _mm_shuffle_ps(qx_q2, qx_q2, 0x00);
  __m128 r1_diag = _mm_sub_ps(one, _mm_add_ps(xx, zz));
  
  __m128 yz = _mm_shuffle_ps(qy_q2, qy_q2, 0xAA);
  __m128 wx = _mm_shuffle_ps(qx_q2, qx_q2, 0xFF);
  __m128 r1_y = _mm_add_ps(yz, wx); // zy + wx
  
  // Pack Row 1
  __m128 r1_tmp1 = _mm_unpacklo_ps(r1_x, r1_diag);
  __m128 r1_tmp2 = _mm_movelh_ps(r1_y, _mm_setzero_ps());
  __m128 row1 = _mm_shuffle_ps(r1_tmp1, r1_tmp2, _MM_SHUFFLE(2, 0, 1, 0));
  _mm_storeu_ps(&m.m[4], row1);
  
  // --- Row 2: [2xz+2wy, 2yz-2wx, 1-2xx-2yy, 0] ---
  __m128 r2_x = _mm_add_ps(xz, wy); // zx + wy
  __m128 r2_y = _mm_sub_ps(yz, wx); // zy - wx
  __m128 r2_diag = _mm_sub_ps(one, _mm_add_ps(xx, yy));
  
  // Pack Row 2
  __m128 r2_tmp1 = _mm_unpacklo_ps(r2_x, r2_y);
  __m128 r2_tmp2 = _mm_movelh_ps(r2_diag, _mm_setzero_ps());
  __m128 row2 = _mm_shuffle_ps(r2_tmp1, r2_tmp2, _MM_SHUFFLE(2, 0, 1, 0));
  _mm_storeu_ps(&m.m[8], row2);
  
  // --- Row 3: [0, 0, 0, 1] ---
  _mm_storeu_ps(&m.m[12], _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Fused-Splat Optimization
  float32x4_t q_vec = vld1q_f32(q.v);
  float32x4_t q2    = vaddq_f32(q_vec, q_vec);
  float32x4_t one   = vdupq_n_f32(1.0f);
  float32x4_t zero  = vdupq_n_f32(0.0f);

  // Parallel products using fused scalar-multiply
  float32x4_t qx_q2 = vmulq_laneq_f32(q2, q_vec, 0); 
  float32x4_t qy_q2 = vmulq_laneq_f32(q2, q_vec, 1); 
  float32x4_t qz_q2 = vmulq_laneq_f32(q2, q_vec, 2); 

  // --- Row 0 ---
  float32x4_t yy = vdupq_laneq_f32(qy_q2, 1);
  float32x4_t zz = vdupq_laneq_f32(qz_q2, 2);
  float32x4_t r0_diag = vsubq_f32(one, vaddq_f32(yy, zz));
  
  float32x4_t xy = vdupq_laneq_f32(qx_q2, 1);
  float32x4_t wz = vdupq_laneq_f32(qz_q2, 3);
  float32x4_t r0_x = vaddq_f32(xy, wz);
  
  float32x4_t xz = vdupq_laneq_f32(qx_q2, 2);
  float32x4_t wy = vdupq_laneq_f32(qy_q2, 3);
  float32x4_t r0_y = vsubq_f32(xz, wy);
  
  // Pack Row 0: [r0_diag, r0_x, r0_y, 0]
  // Interleave logic: zip1(A,B) -> A0,B0,A1,B1...
  float32x4_t r0_lo = vzip1q_f32(r0_diag, r0_x); // [diag, x, diag, x]
  float32x4_t r0_hi = vzip1q_f32(r0_y, zero);    // [y, 0, y, 0]
  float32x4_t row0  = vcombine_f32(vget_low_f32(r0_lo), vget_low_f32(r0_hi));
  vst1q_f32(&m.m[0], row0);

  // --- Row 1 ---
  float32x4_t r1_x = vsubq_f32(xy, wz);
  
  float32x4_t xx = vdupq_laneq_f32(qx_q2, 0);
  float32x4_t r1_diag = vsubq_f32(one, vaddq_f32(xx, zz));
  
  float32x4_t yz = vdupq_laneq_f32(qy_q2, 2);
  float32x4_t wx = vdupq_laneq_f32(qx_q2, 3);
  float32x4_t r1_y = vaddq_f32(yz, wx);
  
  float32x4_t r1_lo = vzip1q_f32(r1_x, r1_diag);
  float32x4_t r1_hi = vzip1q_f32(r1_y, zero);
  float32x4_t row1  = vcombine_f32(vget_low_f32(r1_lo), vget_low_f32(r1_hi));
  vst1q_f32(&m.m[4], row1);

  // --- Row 2 ---
  float32x4_t r2_x = vaddq_f32(xz, wy);
  float32x4_t r2_y = vsubq_f32(yz, wx);
  float32x4_t r2_diag = vsubq_f32(one, vaddq_f32(xx, yy));
  
  float32x4_t r2_lo = vzip1q_f32(r2_x, r2_y);
  float32x4_t r2_hi = vzip1q_f32(r2_diag, zero);
  float32x4_t row2  = vcombine_f32(vget_low_f32(r2_lo), vget_low_f32(r2_hi));
  vst1q_f32(&m.m[8], row2);

  // --- Row 3 ---
  static const float r3_data[] = {0.0f, 0.0f, 0.0f, 1.0f};
  vst1q_f32(&m.m[12], vld1q_f32(r3_data));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM Shuffle Optimization
  v128_t q_vec = wasm_v128_load(q.v);
  v128_t q2    = wasm_f32x4_add(q_vec, q_vec);
  v128_t one   = wasm_f32x4_splat(1.0f);
  v128_t zero  = wasm_f32x4_splat(0.0f);

  v128_t q_x = wasm_v32x4_shuffle(q_vec, q_vec, 0, 0, 0, 0);
  v128_t q_y = wasm_v32x4_shuffle(q_vec, q_vec, 1, 1, 1, 1);
  v128_t q_z = wasm_v32x4_shuffle(q_vec, q_vec, 2, 2, 2, 2);

  v128_t qx_q2 = wasm_f32x4_mul(q_x, q2);
  v128_t qy_q2 = wasm_f32x4_mul(q_y, q2);
  v128_t qz_q2 = wasm_f32x4_mul(q_z, q2);

  // Component Shuffles
  v128_t yy = wasm_v32x4_shuffle(qy_q2, qy_q2, 1, 1, 1, 1);
  v128_t zz = wasm_v32x4_shuffle(qz_q2, qz_q2, 2, 2, 2, 2);
  v128_t xx = wasm_v32x4_shuffle(qx_q2, qx_q2, 0, 0, 0, 0);

  v128_t xy = wasm_v32x4_shuffle(qx_q2, qx_q2, 1, 1, 1, 1);
  v128_t wz = wasm_v32x4_shuffle(qz_q2, qz_q2, 3, 3, 3, 3);
  
  v128_t xz = wasm_v32x4_shuffle(qx_q2, qx_q2, 2, 2, 2, 2);
  v128_t wy = wasm_v32x4_shuffle(qy_q2, qy_q2, 3, 3, 3, 3);

  v128_t yz = wasm_v32x4_shuffle(qy_q2, qy_q2, 2, 2, 2, 2);
  v128_t wx = wasm_v32x4_shuffle(qx_q2, qx_q2, 3, 3, 3, 3);

  // Row 0
  v128_t r0_diag = wasm_f32x4_sub(one, wasm_f32x4_add(yy, zz));
  v128_t r0_x    = wasm_f32x4_add(xy, wz);
  v128_t r0_y    = wasm_f32x4_sub(xz, wy);
  // Construct [diag, x, y, 0]
  v128_t r0_lo = wasm_v32x4_shuffle(r0_diag, r0_x, 0, 4, 0, 4); 
  v128_t r0_hi = wasm_v32x4_shuffle(r0_y, zero, 0, 4, 0, 4);
  v128_t row0  = wasm_v32x4_shuffle(r0_lo, r0_hi, 0, 1, 4, 5);
  wasm_v128_store(&m.m[0], row0);

  // Row 1
  v128_t r1_x    = wasm_f32x4_sub(xy, wz);
  v128_t r1_diag = wasm_f32x4_sub(one, wasm_f32x4_add(xx, zz));
  v128_t r1_y    = wasm_f32x4_add(yz, wx);
  v128_t r1_lo = wasm_v32x4_shuffle(r1_x, r1_diag, 0, 4, 0, 4);
  v128_t r1_hi = wasm_v32x4_shuffle(r1_y, zero, 0, 4, 0, 4);
  v128_t row1  = wasm_v32x4_shuffle(r1_lo, r1_hi, 0, 1, 4, 5);
  wasm_v128_store(&m.m[4], row1);

  // Row 2
  v128_t r2_x    = wasm_f32x4_add(xz, wy);
  v128_t r2_y    = wasm_f32x4_sub(yz, wx);
  v128_t r2_diag = wasm_f32x4_sub(one, wasm_f32x4_add(xx, yy));
  v128_t r2_lo = wasm_v32x4_shuffle(r2_x, r2_y, 0, 4, 0, 4);
  v128_t r2_hi = wasm_v32x4_shuffle(r2_diag, zero, 0, 4, 0, 4);
  v128_t row2  = wasm_v32x4_shuffle(r2_lo, r2_hi, 0, 1, 4, 5);
  wasm_v128_store(&m.m[8], row2);

  // Row 3
  wasm_v128_store(&m.m[12], wasm_f32x4_make(0.0f, 0.0f, 0.0f, 1.0f));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Gather Optimization
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t q_vec = __riscv_vle32_v_f32m1(q.v, vl);
  vfloat32m1_t q2    = __riscv_vfadd_vv_f32m1(q_vec, q_vec, vl);
  vfloat32m1_t one   = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  vfloat32m1_t zero  = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  vuint32m1_t idx_0 = __riscv_vmv_v_x_u32m1(0, vl);
  vuint32m1_t idx_1 = __riscv_vmv_v_x_u32m1(1, vl);
  vuint32m1_t idx_2 = __riscv_vmv_v_x_u32m1(2, vl);
  vuint32m1_t idx_3 = __riscv_vmv_v_x_u32m1(3, vl);

  vfloat32m1_t q_x = __riscv_vrgather_vv_f32m1(q_vec, idx_0, vl);
  vfloat32m1_t q_y = __riscv_vrgather_vv_f32m1(q_vec, idx_1, vl);
  vfloat32m1_t q_z = __riscv_vrgather_vv_f32m1(q_vec, idx_2, vl);

  vfloat32m1_t qx_q2 = __riscv_vfmul_vv_f32m1(q_x, q2, vl);
  vfloat32m1_t qy_q2 = __riscv_vfmul_vv_f32m1(q_y, q2, vl);
  vfloat32m1_t qz_q2 = __riscv_vfmul_vv_f32m1(q_z, q2, vl);

  // Components via Gather
  vfloat32m1_t xx = __riscv_vrgather_vv_f32m1(qx_q2, idx_0, vl);
  vfloat32m1_t yy = __riscv_vrgather_vv_f32m1(qy_q2, idx_1, vl);
  vfloat32m1_t zz = __riscv_vrgather_vv_f32m1(qz_q2, idx_2, vl);
  
  vfloat32m1_t xy = __riscv_vrgather_vv_f32m1(qx_q2, idx_1, vl);
  vfloat32m1_t wz = __riscv_vrgather_vv_f32m1(qz_q2, idx_3, vl);
  vfloat32m1_t xz = __riscv_vrgather_vv_f32m1(qx_q2, idx_2, vl);
  vfloat32m1_t wy = __riscv_vrgather_vv_f32m1(qy_q2, idx_3, vl);
  vfloat32m1_t yz = __riscv_vrgather_vv_f32m1(qy_q2, idx_2, vl);
  vfloat32m1_t wx = __riscv_vrgather_vv_f32m1(qx_q2, idx_3, vl);

  // Row 0
  vfloat32m1_t r0_diag = __riscv_vfsub_vv_f32m1(one, __riscv_vfadd_vv_f32m1(yy, zz, vl), vl);
  vfloat32m1_t r0_x    = __riscv_vfadd_vv_f32m1(xy, wz, vl);
  vfloat32m1_t r0_y    = __riscv_vfsub_vv_f32m1(xz, wy, vl);
  vfloat32m1_t row0    = zero;
  row0 = __riscv_vslideup_vx_f32m1(row0, r0_y, 2, vl);
  row0 = __riscv_vslideup_vx_f32m1(row0, r0_x, 1, vl);
  row0 = __riscv_vslideup_vx_f32m1(row0, r0_diag, 0, vl);
  __riscv_vse32_v_f32m1(&m.m[0], row0, vl);

  // Row 1
  vfloat32m1_t r1_x    = __riscv_vfsub_vv_f32m1(xy, wz, vl);
  vfloat32m1_t r1_diag = __riscv_vfsub_vv_f32m1(one, __riscv_vfadd_vv_f32m1(xx, zz, vl), vl);
  vfloat32m1_t r1_y    = __riscv_vfadd_vv_f32m1(yz, wx, vl);
  vfloat32m1_t row1    = zero;
  row1 = __riscv_vslideup_vx_f32m1(row1, r1_y, 2, vl);
  row1 = __riscv_vslideup_vx_f32m1(row1, r1_diag, 1, vl);
  row1 = __riscv_vslideup_vx_f32m1(row1, r1_x, 0, vl);
  __riscv_vse32_v_f32m1(&m.m[4], row1, vl);

  // Row 2
  vfloat32m1_t r2_x    = __riscv_vfadd_vv_f32m1(xz, wy, vl);
  vfloat32m1_t r2_y    = __riscv_vfsub_vv_f32m1(yz, wx, vl);
  vfloat32m1_t r2_diag = __riscv_vfsub_vv_f32m1(one, __riscv_vfadd_vv_f32m1(xx, yy, vl), vl);
  vfloat32m1_t row2    = zero;
  row2 = __riscv_vslideup_vx_f32m1(row2, r2_diag, 2, vl);
  row2 = __riscv_vslideup_vx_f32m1(row2, r2_y, 1, vl);
  row2 = __riscv_vslideup_vx_f32m1(row2, r2_x, 0, vl);
  __riscv_vse32_v_f32m1(&m.m[8], row2, vl);

  // Row 3
  static const float r3_data[] = {0.0f, 0.0f, 0.0f, 1.0f};
  __riscv_vse32_v_f32m1(&m.m[12], __riscv_vle32_v_f32m1(r3_data, vl), vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LSX Broadcast Optimization
  __m128 q_vec = __lsx_vld(q.v, 0);
  __m128 q2    = __lsx_vfadd_s(q_vec, q_vec);
  __m128 one   = __lsx_vldi(0x3F80); // 1.0f if imm valid, else repl
  if (1) { float f=1.0f; one = __lsx_vldrepl_w(&f, 0); }
  __m128 zero  = __lsx_vldi(0);

  __m128 q_x = __lsx_vreplve_w(q_vec, 0);
  __m128 q_y = __lsx_vreplve_w(q_vec, 1);
  __m128 q_z = __lsx_vreplve_w(q_vec, 2);

  __m128 qx_q2 = __lsx_vfmul_s(q_x, q2);
  __m128 qy_q2 = __lsx_vfmul_s(q_y, q2);
  __m128 qz_q2 = __lsx_vfmul_s(q_z, q2);

  __m128 xx = __lsx_vreplve_w(qx_q2, 0);
  __m128 yy = __lsx_vreplve_w(qy_q2, 1);
  __m128 zz = __lsx_vreplve_w(qz_q2, 2);

  __m128 xy = __lsx_vreplve_w(qx_q2, 1);
  __m128 wz = __lsx_vreplve_w(qz_q2, 3);
  __m128 xz = __lsx_vreplve_w(qx_q2, 2);
  __m128 wy = __lsx_vreplve_w(qy_q2, 3);
  __m128 yz = __lsx_vreplve_w(qy_q2, 2);
  __m128 wx = __lsx_vreplve_w(qx_q2, 3);

  // Row 0
  __m128 r0_diag = __lsx_vfsub_s(one, __lsx_vfadd_s(yy, zz));
  __m128 r0_x    = __lsx_vfadd_s(xy, wz);
  __m128 r0_y    = __lsx_vfsub_s(xz, wy);
  __m128 r0_lo   = __lsx_vilvl_w(r0_x, r0_diag); // [diag, x, ...]
  __m128 r0_hi   = __lsx_vilvl_w(zero, r0_y);    // [y, 0, ...]
  __lsx_vst(__lsx_vilvl_d(r0_hi, r0_lo), &m.m[0], 0);

  // Row 1
  __m128 r1_x    = __lsx_vfsub_s(xy, wz);
  __m128 r1_diag = __lsx_vfsub_s(one, __lsx_vfadd_s(xx, zz));
  __m128 r1_y    = __lsx_vfadd_s(yz, wx);
  __m128 r1_lo   = __lsx_vilvl_w(r1_diag, r1_x);
  __m128 r1_hi   = __lsx_vilvl_w(zero, r1_y);
  __lsx_vst(__lsx_vilvl_d(r1_hi, r1_lo), &m.m[4], 0);

  // Row 2
  __m128 r2_x    = __lsx_vfadd_s(xz, wy);
  __m128 r2_y    = __lsx_vfsub_s(yz, wx);
  __m128 r2_diag = __lsx_vfsub_s(one, __lsx_vfadd_s(xx, yy));
  __m128 r2_lo   = __lsx_vilvl_w(r2_y, r2_x);
  __m128 r2_hi   = __lsx_vilvl_w(zero, r2_diag);
  __lsx_vst(__lsx_vilvl_d(r2_hi, r2_lo), &m.m[8], 0);

  // Row 3
  static const float r3_data[] = {0.0f, 0.0f, 0.0f, 1.0f};
  __lsx_vst(__lsx_vld(r3_data, 0), &m.m[12], 0);

#else
  // Scalar fallback
  float x = q.v[0], y = q.v[1], z = q.v[2], w = q.v[3];
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2;
  float zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;

  m.m[0] = 1.0f - yy - zz;
  m.m[1] = xy + wz;
  m.m[2] = xz - wy;
  m.m[3] = 0.0f;

  m.m[4] = xy - wz;
  m.m[5] = 1.0f - xx - zz;
  m.m[6] = yz + wx;
  m.m[7] = 0.0f;

  m.m[8] = xz + wy;
  m.m[9] = yz - wx;
  m.m[10] = 1.0f - xx - yy;
  m.m[11] = 0.0f;

  m.m[12] = 0.0f; m.m[13] = 0.0f; m.m[14] = 0.0f; m.m[15] = 1.0f;
#endif

  return m;
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_axis_angle)(WMATH_TYPE(Vec3) axis,
                                  const float angle_in_radians) {
  const float a = angle_in_radians * 0.5f;
  const float a_s = sinf(a);
  const float a_c = cosf(a);

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64

  // 使用SSE优化实现
  __m128 axis_vec = wcn_load_vec3_partial(axis.v);   // 加载axis的x,y,z分量
  __m128 sin_vec = _mm_set1_ps(a_s);                 // 广播sin值到所有元素
  __m128 result_vec = _mm_mul_ps(axis_vec, sin_vec); // 计算x,y,z分量

  WMATH_TYPE(Quat) result;
  wcn_store_vec3_partial(result.v, result_vec); // 存储x,y,z分量
  result.v[3] = a_c;                            // 设置w分量

  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64

  // 使用NEON优化实现
  float32x4_t axis_vec = wcn_load_vec3_partial(axis.v);  // 加载axis的x,y,z分量
  float32x4_t sin_vec = vdupq_n_f32(a_s);                // 广播sin值到所有元素
  float32x4_t result_vec = vmulq_f32(axis_vec, sin_vec); // 计算x,y,z分量

  WMATH_TYPE(Quat) result;
  wcn_store_vec3_partial(result.v, result_vec); // 存储x,y,z分量
  result.v[3] = a_c;                            // 设置w分量

  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t axis_vec = wcn_load_vec3_partial(axis.v);   // Load axis x,y,z components
  v128_t sin_vec = wasm_f32x4_splat(a_s);            // Broadcast sin value
  v128_t result_vec = wasm_f32x4_mul(axis_vec, sin_vec); // Calculate x,y,z components

  WMATH_TYPE(Quat) result;
  wcn_store_vec3_partial(result.v, result_vec); // Store x,y,z components
  result.v[3] = a_c;                            // Set w component

  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t axis_vec = wcn_load_vec3_partial(axis.v);   // Load axis x,y,z components
  vfloat32m1_t sin_vec = __riscv_vfmv_v_f_f32m1(a_s, 4);  // Broadcast sin value
  vfloat32m1_t result_vec = __riscv_vfmul_vv_f32m1(axis_vec, sin_vec, 4); // Calculate x,y,z components

  WMATH_TYPE(Quat) result;
  wcn_store_vec3_partial(result.v, result_vec); // Store x,y,z components
  result.v[3] = a_c;                            // Set w component

  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 axis_vec = wcn_load_vec3_partial(axis.v);   // Load axis x,y,z components
  __m128 sin_vec = __lsx_vldrepl_w(&a_s, 0);         // Broadcast sin value
  __m128 result_vec = __lsx_vfmul_s(axis_vec, sin_vec); // Calculate x,y,z components

  WMATH_TYPE(Quat) result;
  wcn_store_vec3_partial(result.v, result_vec); // Store x,y,z components
  result.v[3] = a_c;                            // Set w component

  return result;

#else

  // 原始标量实现
  return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
      .v_x = a_s * axis.v[0],
      .v_y = a_s * axis.v[1],
      .v_z = a_s * axis.v[2],
      .v_w = a_c,
  });

#endif
}

WCN_Math_Vec3_WithAngleAxis WMATH_CALL(Quat,
                                       to_axis_angle)(WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Vec3)
  vec3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = 0,
      .v_y = 0,
      .v_z = 0,
  });
  float angle = acosf(q.v[3]) * 2.0f;
  float s = sinf(angle * 0.5f);
  float ep = wcn_math_get_epsilon();
  if (s > ep) {
    vec3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
        .v_x = q.v[0] / s,
        .v_y = q.v[1] / s,
        .v_z = q.v[2] / s,
    });
  } else {
    vec3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
        .v_x = 1.0f,
        .v_y = 0.0f,
        .v_z = 0.0f,
    });
  }
  return (WCN_Math_Vec3_WithAngleAxis){
      .angle = angle,
      .axis = vec3,
  };
}

WMATH_TYPE(Vec2)
WMATH_CALL(Vec2, transform_mat4)(WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // -----------------------------------------------------------------
  // x86_64 SSE/AVX Implementation
  // Strategy: Linear Combination: res = Col3 + (Col0 * x) + (Col1 * y)
  // -----------------------------------------------------------------
  
  // 1. Safe Load Vec2 (8 bytes) into XMM
  // _mm_load_sd loads a double (64 bits), effectively loading 2 floats without over-reading.
  __m128 vec_v = _mm_castpd_ps(_mm_load_sd((const double*)v.v)); // [x, y, 0, 0]

  // 2. Broadcast x and y
  // x_splat = [x, x, x, x]
  __m128 x_splat = _mm_shuffle_ps(vec_v, vec_v, 0x00);
  // y_splat = [y, y, y, y]
  __m128 y_splat = _mm_shuffle_ps(vec_v, vec_v, 0x55);

  // 3. Load Matrix Columns
  __m128 col0 = _mm_loadu_ps(&m.m[0]);
  __m128 col1 = _mm_loadu_ps(&m.m[4]);
  __m128 col3 = _mm_loadu_ps(&m.m[12]); // Translation

  // 4. Compute: Col3 + Col0*x + Col1*y
#if defined(WCN_HAS_FMA)
  // Use Fused Multiply-Add if available (latency 4-5 cycles vs 8-10 for mul+add)
  __m128 res = _mm_fmadd_ps(col0, x_splat, col3);
  res = _mm_fmadd_ps(col1, y_splat, res);
#else
  __m128 res = _mm_add_ps(col3, _mm_mul_ps(col0, x_splat));
  res = _mm_add_ps(res, _mm_mul_ps(col1, y_splat));
#endif

  // 5. Safe Store (8 bytes)
  _mm_store_sd((double*)vec2.v, _mm_castps_pd(res));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // -----------------------------------------------------------------
  // ARM64 NEON Implementation
  // Optimization: Use VFMA (Fused Multiply Accumulate) by Scalar Lane
  // -----------------------------------------------------------------

  // 1. Safe Load Vec2
  float32x2_t v_small = vld1_f32(v.v); // Loads [x, y] strictly
  
  // 2. Load Matrix Columns
  float32x4_t col0 = vld1q_f32(&m.m[0]);
  float32x4_t col1 = vld1q_f32(&m.m[4]);
  float32x4_t col3 = vld1q_f32(&m.m[12]);

  // 3. Compute: Col3 + Col0*v[0] + Col1*v[1]
  // vfmaq_lane_f32 multiplies vector by a scalar extracted from a vector index
  // res = col3 + col0 * v_small[0]
  float32x4_t res = vfmaq_lane_f32(col3, col0, v_small, 0);
  // res = res + col1 * v_small[1]
  res = vfmaq_lane_f32(res, col1, v_small, 1);

  // 4. Safe Store (low 64 bits)
  vst1_f32(vec2.v, vget_low_f32(res));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // -----------------------------------------------------------------
  // WebAssembly SIMD Implementation
  // -----------------------------------------------------------------

  // 1. Safe Load
  v128_t vec_v = wasm_v128_load64_zero(v.v); // Load 2 floats, zero rest

  // 2. Broadcast
  v128_t x_splat = wasm_v32x4_shuffle(vec_v, vec_v, 0, 0, 0, 0);
  v128_t y_splat = wasm_v32x4_shuffle(vec_v, vec_v, 1, 1, 1, 1);

  // 3. Load Cols
  v128_t col0 = wasm_v128_load(&m.m[0]);
  v128_t col1 = wasm_v128_load(&m.m[4]);
  v128_t col3 = wasm_v128_load(&m.m[12]);

  // 4. Compute
  v128_t t0 = wasm_f32x4_mul(col0, x_splat);
  v128_t t1 = wasm_f32x4_mul(col1, y_splat);
  v128_t res = wasm_f32x4_add(col3, wasm_f32x4_add(t0, t1));

  // 5. Store low 64 bits
  wasm_v128_store64_lane(vec2.v, res, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // -----------------------------------------------------------------
  // RISC-V Vector Implementation
  // -----------------------------------------------------------------
  size_t vl = __riscv_vsetvl_e32m1(4);

  // 1. Load data
  float x = v.v[0];
  float y = v.v[1];
  
  vfloat32m1_t col0 = __riscv_vle32_v_f32m1(&m.m[0], vl);
  vfloat32m1_t col1 = __riscv_vle32_v_f32m1(&m.m[4], vl);
  vfloat32m1_t col3 = __riscv_vle32_v_f32m1(&m.m[12], vl);

  // 2. Compute: vfmacc (Vector Fused Multiply Accumulate)
  // res = col3 + col0 * x
  vfloat32m1_t res = __riscv_vfmacc_vf_f32m1(col3, x, col0, vl);
  // res = res + col1 * y
  res = __riscv_vfmacc_vf_f32m1(res, y, col1, vl);

  // 3. Store (Standard vector store writes more than we need, scalar store is safer/cleaner for Vec2)
  // Extracting first 2 elements is cheap.
  vec2.v[0] = __riscv_vfmv_f_s_f32m1_f32(res);
  vec2.v[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(res, 1, vl));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // -----------------------------------------------------------------
  // LoongArch LSX Implementation
  // -----------------------------------------------------------------
  // Load using replve to broadcast directly from memory is not standard, load then broadcast.
  
  // 1. Load inputs
  // Safe load 64-bit (2 floats) is tricky in pure LSX intrinsics without over-read. 
  // We'll load scalar to be safe or assume padding. Let's use scalar load + repl.
  __m128 x_splat = __lsx_vreplfr2vr_s(v.v[0]);
  __m128 y_splat = __lsx_vreplfr2vr_s(v.v[1]);

  __m128 col0 = __lsx_vld(&m.m[0], 0);
  __m128 col1 = __lsx_vld(&m.m[4], 0);
  __m128 col3 = __lsx_vld(&m.m[12], 0);

  // 2. Compute (fmadd: d = a * b + c)
  __m128 res = __lsx_vfmadd_s(col0, x_splat, col3); // res = col0*x + col3
  res = __lsx_vfmadd_s(col1, y_splat, res);         // res = col1*y + res

  // 3. Store
  __lsx_vstelm_w(res, vec2.v, 0, 0);
  __lsx_vstelm_w(res, vec2.v, 4, 1); // Offset is bytes

#else
  // Scalar Fallback
  float x = v.v[0];
  float y = v.v[1];
  vec2.v[0] = x * m.m[0] + y * m.m[4] + m.m[12];
  vec2.v[1] = x * m.m[1] + y * m.m[5] + m.m[13];
#endif

  return vec2;
}

WMATH_TYPE(Vec2)
WMATH_CALL(Vec2, transform_mat3)(const WMATH_TYPE(Vec2) v, const WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec2) vec2;
  float x = v.v[0];
  float y = v.v[1];
  vec2.v[0] = x * m.m[0] + y * m.m[4] + m.m[8];
  vec2.v[1] = x * m.m[1] + y * m.m[5] + m.m[9];
  return vec2;
}

WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat4)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) r;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // -----------------------------------------------------------------
  // x86_64 SSE/AVX Implementation
  // -----------------------------------------------------------------

  // 1. 安全加载 Vec3 (12 bytes)
  // 我们不能直接 loadu_ps，因为可能越界。
  // 方法：加载 X 和 Y (64位)，然后加载 Z (32位)
  __m128 xy = _mm_castpd_ps(_mm_load_sd((const double*)&v.v[0]));
  __m128 z_val = _mm_load_ss(&v.v[2]);
  // 将 Z 移动到正确位置: xy=[x, y, 0, 0], z_val=[z, 0, 0, 0] -> vec=[x, y, z, 0]
  __m128 vec_v = _mm_movelh_ps(xy, z_val);

  // 2. 加载矩阵列
  // Mat4 保证是 16 字节对齐且完整的，可以直接加载
  __m128 col0 = _mm_loadu_ps(&m.m[0]);
  __m128 col1 = _mm_loadu_ps(&m.m[4]);
  __m128 col2 = _mm_loadu_ps(&m.m[8]);
  __m128 col3 = _mm_loadu_ps(&m.m[12]); // Translation column

  // 3. 广播分量
  __m128 x = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 y = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 z = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(2, 2, 2, 2));

  // 4. 线性组合: res = Col3 + x*Col0 + y*Col1 + z*Col2
#if defined(WCN_HAS_FMA)
  __m128 res = _mm_fmadd_ps(col0, x, col3);
  res = _mm_fmadd_ps(col1, y, res);
  res = _mm_fmadd_ps(col2, z, res);
#else
  __m128 res = _mm_add_ps(col3, _mm_mul_ps(col0, x));
  res = _mm_add_ps(res, _mm_mul_ps(col1, y));
  res = _mm_add_ps(res, _mm_mul_ps(col2, z));
#endif

  // 5. 透视除法 (Perspective Divide)
  // 提取 w 分量
  float w = _mm_cvtss_f32(_mm_shuffle_ps(res, res, _MM_SHUFFLE(3, 3, 3, 3)));

  // 避免除以零 (极小值处理)
  if (fabsf(w) < 1e-6f) w = 1.0f;

  // 执行除法
  res = _mm_div_ps(res, _mm_set1_ps(w));

  // 6. 安全存储 Vec3 (12 bytes)
  // 存储低位 64 bit (x, y)
  _mm_store_sd((double*)&r.v[0], _mm_castps_pd(res));
  // 存储高位 Z (将 Z 移到低位然后 store_ss)
  __m128 res_z = _mm_movehl_ps(res, res);
  _mm_store_ss(&r.v[2], res_z);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // -----------------------------------------------------------------
  // ARM64 NEON Implementation
  // -----------------------------------------------------------------

  // 1. 安全加载
  // vld1_f32 加载 2 个 float (64位)，正好读取 x, y
  float32x2_t v_xy = vld1_f32(v.v);
  // 单独加载 z
  float z_val = v.v[2];

  // 2. 加载矩阵列
  float32x4_t col0 = vld1q_f32(&m.m[0]);
  float32x4_t col1 = vld1q_f32(&m.m[4]);
  float32x4_t col2 = vld1q_f32(&m.m[8]);
  float32x4_t col3 = vld1q_f32(&m.m[12]);

  // 3. 计算 (使用 Lane 乘加指令，极其高效)
  // res = col3 + col0 * v.x
  float32x4_t res = vfmaq_lane_f32(col3, col0, v_xy, 0);
  // res += col1 * v.y
  res = vfmaq_lane_f32(res, col1, v_xy, 1);
  // res += col2 * v.z (这里直接用标量乘法指令)
  res = vmlaq_n_f32(res, col2, z_val);

  // 4. 透视除法
  float w = vgetq_lane_f32(res, 3);
  if (fabsf(w) < 1e-6f) w = 1.0f;

  // 这种除法在 ARM 上通常很快
  res = vmulq_n_f32(res, 1.0f / w);

  // 5. 安全存储
  vst1_f32(r.v, vget_low_f32(res)); // 存储 x, y
  vst1q_lane_f32(&r.v[2], res, 2);  // 存储 z

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // -----------------------------------------------------------------
  // WebAssembly SIMD Implementation
  // -----------------------------------------------------------------

  // 1. 安全加载
  v128_t xy = wasm_v128_load64_zero(v.v); // 加载 8 字节 (x, y)，高位清零
  v128_t vec_v = wasm_f32x4_replace_lane(xy, 2, v.v[2]); // 插入 z

  // 2. 加载列
  v128_t col0 = wasm_v128_load(&m.m[0]);
  v128_t col1 = wasm_v128_load(&m.m[4]);
  v128_t col2 = wasm_v128_load(&m.m[8]);
  v128_t col3 = wasm_v128_load(&m.m[12]);

  // 3. 广播并计算
  v128_t x = wasm_i32x4_shuffle(vec_v, vec_v, 0, 0, 0, 0);
  v128_t y = wasm_i32x4_shuffle(vec_v, vec_v, 1, 1, 1, 1);
  v128_t z = wasm_i32x4_shuffle(vec_v, vec_v, 2, 2, 2, 2);

  v128_t res = wasm_f32x4_add(col3, wasm_f32x4_mul(col0, x));
  res = wasm_f32x4_add(res, wasm_f32x4_mul(col1, y));
  res = wasm_f32x4_add(res, wasm_f32x4_mul(col2, z));

  // 4. 透视除法
  float w = wasm_f32x4_extract_lane(res, 3);
  if (fabsf(w) < 1e-6f) w = 1.0f;

  res = wasm_f32x4_mul(res, wasm_f32x4_splat(1.0f / w));

  // 5. 安全存储
  wasm_v128_store64_lane(r.v, res, 0); // Store x, y
  wasm_v128_store32_lane(&r.v[2], res, 2); // Store z

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // -----------------------------------------------------------------
  // RISC-V Vector Implementation
  // -----------------------------------------------------------------
  size_t vl = 4; // 处理 4 个 float

  // 1. 标量加载 Vec3 (RVV 处理非对齐或部分加载比较繁琐，标量加载更快)
  float x_val = v.v[0];
  float y_val = v.v[1];
  float z_val = v.v[2];

  // 2. 加载列
  vfloat32m1_t col0 = __riscv_vle32_v_f32m1(&m.m[0], vl);
  vfloat32m1_t col1 = __riscv_vle32_v_f32m1(&m.m[4], vl);
  vfloat32m1_t col2 = __riscv_vle32_v_f32m1(&m.m[8], vl);
  vfloat32m1_t col3 = __riscv_vle32_v_f32m1(&m.m[12], vl);

  // 3. 计算 (使用 vfmacc.vf 向量-标量乘加)
  // res = col3 + col0 * x
  vfloat32m1_t res = __riscv_vfmacc_vf_f32m1(col3, x_val, col0, vl);
  // res += col1 * y
  res = __riscv_vfmacc_vf_f32m1(res, y_val, col1, vl);
  // res += col2 * z
  res = __riscv_vfmacc_vf_f32m1(res, z_val, col2, vl);

  // 4. 透视除法
  float w = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(res, res, 3, vl));
  if (fabsf(w) < 1e-6f) w = 1.0f;

  res = __riscv_vfmul_vf_f32m1(res, 1.0f / w, vl);

  // 5. 存储
  // 提取回标量存储最安全
  r.v[0] = __riscv_vfmv_f_s_f32m1_f32(res);
  r.v[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(res, res, 1, vl));
  r.v[2] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(res, res, 2, vl));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // -----------------------------------------------------------------
  // LoongArch LSX Implementation
  // -----------------------------------------------------------------

  // 1. 广播标量
  __m128 x = __lsx_vreplfr2vr_s(v.v[0]);
  __m128 y = __lsx_vreplfr2vr_s(v.v[1]);
  __m128 z = __lsx_vreplfr2vr_s(v.v[2]);

  // 2. 加载列
  __m128 col0 = __lsx_vld(&m.m[0], 0);
  __m128 col1 = __lsx_vld(&m.m[4], 0);
  __m128 col2 = __lsx_vld(&m.m[8], 0);
  __m128 col3 = __lsx_vld(&m.m[12], 0);

  // 3. 计算
  // res = col3 + col0 * x
  __m128 res = __lsx_vfmadd_s(col0, x, col3);
  res = __lsx_vfmadd_s(col1, y, res);
  res = __lsx_vfmadd_s(col2, z, res);

  // 4. 透视除法
  float w = __lsx_vpickve2gr_w(res, 3); // 提取 float 的位作为 int，需要 bitcast
  // 或者使用 store 提取
  float w_val;
  __lsx_vstelm_w(res, &w_val, 0, 3);

  if (fabsf(w_val) < 1e-6f) w_val = 1.0f;
  res = __lsx_vfmul_s(res, __lsx_vreplfr2vr_s(1.0f / w_val));

  // 5. 存储
  __lsx_vstelm_w(res, r.v, 0, 0);
  __lsx_vstelm_w(res, r.v, 4, 1);
  __lsx_vstelm_w(res, r.v, 8, 2);

#else
  // Scalar Fallback
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];

  // 计算 w (Dot Product of Row 3)
  // m[3], m[7], m[11], m[15] 是第4行（如果是列主序，则是各列的第4个元素）
  float w = x * m.m[3] + y * m.m[7] + z * m.m[11] + m.m[15];

  if (fabsf(w) < 1e-6f) w = 1.0f;
  float inv_w = 1.0f / w;

  // Linear Combination 展开形式
  r.v[0] = (x * m.m[0] + y * m.m[4] + z * m.m[8] + m.m[12]) * inv_w;
  r.v[1] = (x * m.m[1] + y * m.m[5] + z * m.m[9] + m.m[13]) * inv_w;
  r.v[2] = (x * m.m[2] + y * m.m[6] + z * m.m[10] + m.m[14]) * inv_w;
#endif

  return r;
}

// vec3 transformMat4Upper3x3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat4_upper3x3)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) vec3;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // -----------------------------------------------------------------
  // x86_64 SSE/AVX
  // Method: Linear Combination (x*Col0 + y*Col1 + z*Col2)
  // -----------------------------------------------------------------

  // 1. Safe Load Vec3 (12 bytes)
  // Load x, y as double (low 64 bits), z as scalar (high 32 bits of low 64)
  __m128 xy = _mm_castpd_ps(_mm_load_sd((const double*)v.v)); 
  __m128 z_val = _mm_load_ss(&v.v[2]);
  // Combine: [x, y, z, 0]
  __m128 vec_v = _mm_movelh_ps(xy, z_val);

  // 2. Broadcast
  __m128 x_splat = _mm_shuffle_ps(vec_v, vec_v, 0x00);
  __m128 y_splat = _mm_shuffle_ps(vec_v, vec_v, 0x55);
  __m128 z_splat = _mm_shuffle_ps(z_val, z_val, 0x00);

  // 3. Load Columns (Ignore Col3/Translation)
  __m128 col0 = _mm_loadu_ps(&m.m[0]);
  __m128 col1 = _mm_loadu_ps(&m.m[4]);
  __m128 col2 = _mm_loadu_ps(&m.m[8]);

  // 4. Compute Linear Combination
#if defined(WCN_HAS_FMA)
  __m128 res = _mm_mul_ps(col0, x_splat);
  res = _mm_fmadd_ps(col1, y_splat, res);
  res = _mm_fmadd_ps(col2, z_splat, res);
#else
  __m128 res = _mm_mul_ps(col0, x_splat);
  res = _mm_add_ps(res, _mm_mul_ps(col1, y_splat));
  res = _mm_add_ps(res, _mm_mul_ps(col2, z_splat));
#endif

  // 5. Safe Store (12 bytes)
  // Store low 2 floats (x, y)
  _mm_store_sd((double*)&vec3.v[0], _mm_castps_pd(res));
  // Store 3rd float (z) via shuffle and scalar store
  __m128 res_z = _mm_movehl_ps(res, res);
  _mm_store_ss(&vec3.v[2], res_z);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // -----------------------------------------------------------------
  // ARM64 NEON
  // Method: VFMA with Lane Broadcasting
  // -----------------------------------------------------------------

  // 1. Safe Load
  float32x2_t v_xy = vld1_f32(v.v);       // Load x, y
  float32x4_t v_z_vec = vld1q_dup_f32(&v.v[2]); // Load z (splatted is fine)
  
  // Create a combined vector [x, y, x, y] for lane access
  float32x4_t v_xy_vec = vcombine_f32(v_xy, v_xy);

  // 2. Load Columns
  float32x4_t col0 = vld1q_f32(&m.m[0]);
  float32x4_t col1 = vld1q_f32(&m.m[4]);
  float32x4_t col2 = vld1q_f32(&m.m[8]);

  // 3. Compute
  // res = Col0 * x
  float32x4_t res = vmulq_lane_f32(col0, vget_low_f32(v_xy_vec), 0);
  // res += Col1 * y
  res = vfmaq_lane_f32(res, col1, vget_low_f32(v_xy_vec), 1);
  // res += Col2 * z
  res = vfmaq_f32(res, col2, v_z_vec);

  // 4. Store (Ignore w component)
  vst1_f32(vec3.v, vget_low_f32(res)); // Store x, y
  vst1q_lane_f32(&vec3.v[2], res, 2);  // Store z

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // -----------------------------------------------------------------
  // WebAssembly SIMD
  // -----------------------------------------------------------------

  // 1. Safe Load
  v128_t v_xy = wasm_v128_load64_zero(v.v); // x, y, 0, 0
  v128_t vec_v = wasm_f32x4_replace_lane(v_xy, 2, v.v[2]); // x, y, z, 0

  // 2. Broadcast
  v128_t x_splat = wasm_v32x4_shuffle(vec_v, vec_v, 0, 0, 0, 0);
  v128_t y_splat = wasm_v32x4_shuffle(vec_v, vec_v, 1, 1, 1, 1);
  v128_t z_splat = wasm_v32x4_shuffle(vec_v, vec_v, 2, 2, 2, 2);

  // 3. Load Columns
  v128_t col0 = wasm_v128_load(&m.m[0]);
  v128_t col1 = wasm_v128_load(&m.m[4]);
  v128_t col2 = wasm_v128_load(&m.m[8]);

  // 4. Compute
  v128_t res = wasm_f32x4_mul(col0, x_splat);
  res = wasm_f32x4_add(res, wasm_f32x4_mul(col1, y_splat));
  res = wasm_f32x4_add(res, wasm_f32x4_mul(col2, z_splat));

  // 5. Safe Store
  wasm_v128_store64_lane(vec3.v, res, 0); // Store x, y
  vec3.v[2] = wasm_f32x4_extract_lane(res, 2); // Store z

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // -----------------------------------------------------------------
  // RISC-V Vector
  // -----------------------------------------------------------------
  size_t vl = __riscv_vsetvl_e32m1(4);

  // 1. Load Data
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];

  vfloat32m1_t col0 = __riscv_vle32_v_f32m1(&m.m[0], vl);
  vfloat32m1_t col1 = __riscv_vle32_v_f32m1(&m.m[4], vl);
  vfloat32m1_t col2 = __riscv_vle32_v_f32m1(&m.m[8], vl);

  // 2. Compute
  vfloat32m1_t res = __riscv_vfmul_vf_f32m1(col0, x, vl);
  res = __riscv_vfmacc_vf_f32m1(res, y, col1, vl);
  res = __riscv_vfmacc_vf_f32m1(res, z, col2, vl);

  // 3. Store (3 elements)
  // vslidedown/extract is safest portable way if tu policy obscure
  vec3.v[0] = __riscv_vfmv_f_s_f32m1_f32(res);
  vec3.v[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(res, 1, vl));
  vec3.v[2] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(res, 2, vl));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // -----------------------------------------------------------------
  // LoongArch LSX
  // -----------------------------------------------------------------
  __m128 x_splat = __lsx_vreplfr2vr_s(v.v[0]);
  __m128 y_splat = __lsx_vreplfr2vr_s(v.v[1]);
  __m128 z_splat = __lsx_vreplfr2vr_s(v.v[2]);

  __m128 col0 = __lsx_vld(&m.m[0], 0);
  __m128 col1 = __lsx_vld(&m.m[4], 0);
  __m128 col2 = __lsx_vld(&m.m[8], 0);

  __m128 res = __lsx_vfmul_s(col0, x_splat);
  res = __lsx_vfmadd_s(col1, y_splat, res);
  res = __lsx_vfmadd_s(col2, z_splat, res);

  __lsx_vstelm_w(res, vec3.v, 0, 0);
  __lsx_vstelm_w(res, vec3.v, 4, 1);
  __lsx_vstelm_w(res, vec3.v, 8, 2);

#else
  // Scalar Fallback
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];

  vec3.v[0] = x * m.m[0] + y * m.m[4] + z * m.m[8];
  vec3.v[1] = x * m.m[1] + y * m.m[5] + z * m.m[9];
  vec3.v[2] = x * m.m[2] + y * m.m[6] + z * m.m[10];
#endif

  return vec3;
}

// vec3 transformMat3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat3)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec3) r;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE Implementation
  // Load vector v (x, y, z, ?)
  __m128 vec_v = wcn_load_vec3_partial(v.v);

  // Load matrix columns directly (Matching scalar stride 0, 4, 8)
  // Assuming Mat3 is padded to 16-byte alignment (stride 4 floats)
  __m128 col0 = _mm_loadu_ps(&m.m[0]);
  __m128 col1 = _mm_loadu_ps(&m.m[4]);
  __m128 col2 = _mm_loadu_ps(&m.m[8]);

  // Broadcast v.x, v.y, v.z
  __m128 x = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 y = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 z = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(2, 2, 2, 2));

  // Result = x*col0 + y*col1 + z*col2
  // Using FMA if available would be better, but basic mul/add is standard SSE2
  __m128 res = _mm_mul_ps(col0, x);
  #if defined(WCN_HAS_FMA)
    res = _mm_fmadd_ps(col1, y, res);
    res = _mm_fmadd_ps(col2, z, res);
  #else
    res = _mm_add_ps(res, _mm_mul_ps(col1, y));
    res = _mm_add_ps(res, _mm_mul_ps(col2, z));
  #endif

  wcn_store_vec3_partial(r.v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Implementation
  float32x4_t vec_v = wcn_load_vec3_partial(v.v);

  float32x4_t col0 = vld1q_f32(&m.m[0]);
  float32x4_t col1 = vld1q_f32(&m.m[4]);
  float32x4_t col2 = vld1q_f32(&m.m[8]);

  // Multiply Accumulate by Element (FMA with scalar broadcast)
  // res = col0 * v.x
  float32x4_t res = vmulq_laneq_f32(col0, vec_v, 0);
  // res += col1 * v.y
  res = vfmaq_laneq_f32(res, col1, vec_v, 1);
  // res += col2 * v.z
  res = vfmaq_laneq_f32(res, col2, vec_v, 2);

  wcn_store_vec3_partial(r.v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD Implementation
  v128_t vec_v = wcn_load_vec3_partial(v.v);

  v128_t col0 = wasm_v128_load(&m.m[0]);
  v128_t col1 = wasm_v128_load(&m.m[4]);
  v128_t col2 = wasm_v128_load(&m.m[8]);

  // Splat components
  v128_t x = wasm_i32x4_shuffle(vec_v, vec_v, 0, 0, 0, 0);
  v128_t y = wasm_i32x4_shuffle(vec_v, vec_v, 1, 1, 1, 1);
  v128_t z = wasm_i32x4_shuffle(vec_v, vec_v, 2, 2, 2, 2);

  // Linear combination
  v128_t res = wasm_f32x4_mul(col0, x);
  res = wasm_f32x4_add(res, wasm_f32x4_mul(col1, y));
  res = wasm_f32x4_add(res, wasm_f32x4_mul(col2, z));

  wcn_store_vec3_partial(r.v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Implementation (RVV 1.0)
  size_t vl = 4; // Assuming operation on 4 floats
  vfloat32m1_t vec_v = wcn_load_vec3_partial(v.v);
  
  // Load columns
  vfloat32m1_t col0 = __riscv_vle32_v_f32m1(&m.m[0], vl);
  vfloat32m1_t col1 = __riscv_vle32_v_f32m1(&m.m[4], vl);
  vfloat32m1_t col2 = __riscv_vle32_v_f32m1(&m.m[8], vl);

  // Extract scalars (0, 1, 2)
  float x_val = __riscv_vfmv_f_s_f32m1_f32(vec_v);
  float y_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(vec_v, vec_v, 1, vl));
  float z_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(vec_v, vec_v, 2, vl));

  // r = col0 * x
  vfloat32m1_t res = __riscv_vfmul_vf_f32m1(col0, x_val, vl);
  // r += col1 * y
  res = __riscv_vfmacc_vf_f32m1(res, y_val, col1, vl);
  // r += col2 * z
  res = __riscv_vfmacc_vf_f32m1(res, z_val, col2, vl);

  wcn_store_vec3_partial(r.v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Implementation
  __m128 vec_v = wcn_load_vec3_partial(v.v);

  __m128 col0 = __lsx_vld(&m.m[0], 0);
  __m128 col1 = __lsx_vld(&m.m[4], 0);
  __m128 col2 = __lsx_vld(&m.m[8], 0);

  // Broadcast components
  __m128 x = __lsx_vreplve_w(vec_v, 0);
  __m128 y = __lsx_vreplve_w(vec_v, 1);
  __m128 z = __lsx_vreplve_w(vec_v, 2);

  // FMADD: d = a * b + c
  // res = col0 * x
  __m128 res = __lsx_vfmul_s(col0, x);
  // res = col1 * y + res
  res = __lsx_vfmadd_s(col1, y, res);
  // res = col2 * z + res
  res = __lsx_vfmadd_s(col2, z, res);

  wcn_store_vec3_partial(r.v, res);

#else
  // Scalar fallback
  // Logic matches: r = x * Col0 + y * Col1 + z * Col2
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];
  
  // Unrolled Linear Combination
  r.v[0] = x * m.m[0] + y * m.m[4] + z * m.m[8];
  r.v[1] = x * m.m[1] + y * m.m[5] + z * m.m[9];
  r.v[2] = x * m.m[2] + y * m.m[6] + z * m.m[10];
#endif

  return r;
}

// vec3 transformQuat
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_quat)
(WMATH_TYPE(Vec3) v, WMATH_TYPE(Quat) q) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64

  // SSE -- 修正：不要预先把 q.w * 2
  __m128 v_vec = wcn_load_vec3_partial(v.v); // [x, y, z, 0]
  __m128 q_vec = _mm_loadu_ps(q.v);          // [qx, qy, qz, qw]

  __m128 q_yzx = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 q_zxy = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(3, 1, 0, 2));

  __m128 v_yzx = _mm_shuffle_ps(v_vec, v_vec, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 v_zxy = _mm_shuffle_ps(v_vec, v_vec, _MM_SHUFFLE(3, 1, 0, 2));

  __m128 uv = _mm_sub_ps(_mm_mul_ps(q_yzx, v_zxy), _mm_mul_ps(q_zxy, v_yzx));

  __m128 uv_yzx = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 uv_zxy = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 1, 0, 2));
  __m128 uuv = _mm_sub_ps(_mm_mul_ps(q_yzx, uv_zxy), _mm_mul_ps(q_zxy, uv_yzx));

  // 这里用 q_w（而不是 2*q_w）
  __m128 q_w = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(3, 3, 3, 3));
  __m128 t1 = _mm_mul_ps(q_w, uv);        // q.w * uv
  __m128 t2 = _mm_add_ps(t1, uuv);        // q.w*uv + uuv
  __m128 t3 = _mm_add_ps(t2, t2);         // 2 * (q.w*uv + uuv)
  __m128 res = _mm_add_ps(v_vec, t3);

  WMATH_TYPE(Vec3) r;
  wcn_store_vec3_partial(r.v, res);
  return r;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64

  // NEON -- 同样修正：不用提前乘 2
  float32x4_t v_vec = wcn_load_vec3_partial(v.v);
  float32x4_t q_vec = vld1q_f32(q.v);

  float32x4_t q_yzx = vextq_f32(q_vec, q_vec, 1);
  float32x4_t q_zxy = vextq_f32(q_vec, q_vec, 2);

  float32x4_t v_yzx = vextq_f32(v_vec, v_vec, 1);
  float32x4_t v_zxy = vextq_f32(v_vec, v_vec, 2);

  float32x4_t uv = vsubq_f32(vmulq_f32(q_yzx, v_zxy), vmulq_f32(q_zxy, v_yzx));

  float32x4_t uv_yzx = vextq_f32(uv, uv, 1);
  float32x4_t uv_zxy = vextq_f32(uv, uv, 2);
  float32x4_t uuv = vsubq_f32(vmulq_f32(q_yzx, uv_zxy), vmulq_f32(q_zxy, uv_yzx));

  float32x4_t q_w = vdupq_n_f32(vgetq_lane_f32(q_vec, 3));
  float32x4_t t1 = vmulq_f32(q_w, uv);
  float32x4_t t2 = vaddq_f32(t1, uuv);
  float32x4_t t3 = vaddq_f32(t2, t2);
  float32x4_t res = vaddq_f32(v_vec, t3);

  WMATH_TYPE(Vec3) r;
  wcn_store_vec3_partial(r.v, res);
  return r;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t v_vec = wcn_load_vec3_partial(v.v); // [x, y, z, 0]
  v128_t q_vec = wasm_v128_load(q.v);        // [qx, qy, qz, qw]

  v128_t q_yzx = wasm_f32x4_shuffle(q_vec, q_vec, 3, 0, 2, 1);
  v128_t q_zxy = wasm_f32x4_shuffle(q_vec, q_vec, 3, 1, 0, 2);

  v128_t v_yzx = wasm_f32x4_shuffle(v_vec, v_vec, 3, 0, 2, 1);
  v128_t v_zxy = wasm_f32x4_shuffle(v_vec, v_vec, 3, 1, 0, 2);

  v128_t uv = wasm_f32x4_sub(wasm_f32x4_mul(q_yzx, v_zxy), wasm_f32x4_mul(q_zxy, v_yzx));

  v128_t uv_yzx = wasm_f32x4_shuffle(uv, uv, 3, 0, 2, 1);
  v128_t uv_zxy = wasm_f32x4_shuffle(uv, uv, 3, 1, 0, 2);
  v128_t uuv = wasm_f32x4_sub(wasm_f32x4_mul(q_yzx, uv_zxy), wasm_f32x4_mul(q_zxy, uv_yzx));

  // 这里用 q_w（而不是 2*q_w）
  v128_t q_w = wasm_f32x4_splat(wasm_f32x4_extract_lane(q_vec, 3));
  v128_t t1 = wasm_f32x4_mul(q_w, uv);        // q.w * uv
  v128_t t2 = wasm_f32x4_add(t1, uuv);        // q.w*uv + uuv
  v128_t t3 = wasm_f32x4_add(t2, t2);         // 2 * (q.w*uv + uuv)
  v128_t res = wasm_f32x4_add(v_vec, t3);

  WMATH_TYPE(Vec3) r;
  wcn_store_vec3_partial(r.v, res);
  return r;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float v_temp[4] = {v.v[0], v.v[1], v.v[2], 0.0f};
  vfloat32m1_t v_vec = __riscv_vle32_v_f32m1(v_temp, 4);
  vfloat32m1_t q_vec = __riscv_vle32_v_f32m1(q.v, 4);

  // Create shuffled vectors for quaternion operations
  float q_vals[4] = {__riscv_vfmv_f_s_f32m1(q_vec, 3), __riscv_vfmv_f_s_f32m1(q_vec, 0),
                     __riscv_vfmv_f_s_f32m1(q_vec, 2), __riscv_vfmv_f_s_f32m1(q_vec, 1)};
  vfloat32m1_t q_yzx = __riscv_vle32_v_f32m1(q_vals, 4);

  float q_vals2[4] = {__riscv_vfmv_f_s_f32m1(q_vec, 3), __riscv_vfmv_f_s_f32m1(q_vec, 1),
                      __riscv_vfmv_f_s_f32m1(q_vec, 0), __riscv_vfmv_f_s_f32m1(q_vec, 2)};
  vfloat32m1_t q_zxy = __riscv_vle32_v_f32m1(q_vals2, 4);

  float v_vals[4] = {__riscv_vfmv_f_s_f32m1(v_vec, 3), __riscv_vfmv_f_s_f32m1(v_vec, 0),
                     __riscv_vfmv_f_s_f32m1(v_vec, 2), __riscv_vfmv_f_s_f32m1(v_vec, 1)};
  vfloat32m1_t v_yzx = __riscv_vle32_v_f32m1(v_vals, 4);

  float v_vals2[4] = {__riscv_vfmv_f_s_f32m1(v_vec, 3), __riscv_vfmv_f_s_f32m1(v_vec, 1),
                      __riscv_vfmv_f_s_f32m1(v_vec, 0), __riscv_vfmv_f_s_f32m1(v_vec, 2)};
  vfloat32m1_t v_zxy = __riscv_vle32_v_f32m1(v_vals2, 4);

  vfloat32m1_t uv = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(q_yzx, v_zxy, 4),
                                           __riscv_vfmul_vv_f32m1(q_zxy, v_yzx, 4), 4);

  float uv_vals[4] = {__riscv_vfmv_f_s_f32m1(uv, 3), __riscv_vfmv_f_s_f32m1(uv, 0),
                      __riscv_vfmv_f_s_f32m1(uv, 2), __riscv_vfmv_f_s_f32m1(uv, 1)};
  vfloat32m1_t uv_yzx = __riscv_vle32_v_f32m1(uv_vals, 4);

  float uv_vals2[4] = {__riscv_vfmv_f_s_f32m1(uv, 3), __riscv_vfmv_f_s_f32m1(uv, 1),
                       __riscv_vfmv_f_s_f32m1(uv, 0), __riscv_vfmv_f_s_f32m1(uv, 2)};
  vfloat32m1_t uv_zxy = __riscv_vle32_v_f32m1(uv_vals2, 4);

  vfloat32m1_t uuv = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(q_yzx, uv_zxy, 4),
                                            __riscv_vfmul_vv_f32m1(q_zxy, uv_yzx, 4), 4);

  // 这里用 q_w（而不是 2*q_w）
  vfloat32m1_t q_w = __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1(q_vec, 3), 4);
  vfloat32m1_t t1 = __riscv_vfmul_vv_f32m1(q_w, uv, 4);        // q.w * uv
  vfloat32m1_t t2 = __riscv_vfadd_vv_f32m1(t1, uuv, 4);        // q.w*uv + uuv
  vfloat32m1_t t3 = __riscv_vfadd_vv_f32m1(t2, t2, 4);         // 2 * (q.w*uv + uuv)
  vfloat32m1_t res = __riscv_vfadd_vv_f32m1(v_vec, t3, 4);

  WMATH_TYPE(Vec3) r;
  wcn_store_vec3_partial(r.v, res);
  return r;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 v_vec = wcn_load_vec3_partial(v.v); // [x, y, z, 0]
  __m128 q_vec = __lsx_vld(q.v, 0);          // [qx, qy, qz, qw]

  __m128 q_yzx = __lsx_vshuf4i_w(q_vec, 0x39); // Shuffle to get [qw, qx, qz, qy]
  __m128 q_zxy = __lsx_vshuf4i_w(q_vec, 0x4E); // Shuffle to get [qw, qy, qx, qz]

  __m128 v_yzx = __lsx_vshuf4i_w(v_vec, 0x39);
  __m128 v_zxy = __lsx_vshuf4i_w(v_vec, 0x4E);

  __m128 uv = __lsx_vfsub_s(__lsx_vfmul_s(q_yzx, v_zxy), __lsx_vfmul_s(q_zxy, v_yzx));

  __m128 uv_yzx = __lsx_vshuf4i_w(uv, 0x39);
  __m128 uv_zxy = __lsx_vshuf4i_w(uv, 0x4E);
  __m128 uuv = __lsx_vfsub_s(__lsx_vfmul_s(q_yzx, uv_zxy), __lsx_vfmul_s(q_zxy, uv_yzx));

  // 这里用 q_w（而不是 2*q_w）
  __m128 q_w = __lsx_vreplfr2vr_s(__lsx_vpickve2gr_w(q_vec, 3));
  __m128 t1 = __lsx_vfmul_s(q_w, uv);        // q.w * uv
  __m128 t2 = __lsx_vfadd_s(t1, uuv);        // q.w*uv + uuv
  __m128 t3 = __lsx_vfadd_s(t2, t2);         // 2 * (q.w*uv + uuv)
  __m128 res = __lsx_vfadd_s(v_vec, t3);

  WMATH_TYPE(Vec3) r;
  wcn_store_vec3_partial(r.v, res);
  return r;

#else

  // Scalar (不变)
  WMATH_TYPE(Vec3) r;
  float qx = q.v[0], qy = q.v[1], qz = q.v[2], qw = q.v[3];
  float x = v.v[0], y = v.v[1], z = v.v[2];

  float uvX = qy * z - qz * y;
  float uvY = qz * x - qx * z;
  float uvZ = qx * y - qy * x;

  float uuvX = qy * uvZ - qz * uvY;
  float uuvY = qz * uvX - qx * uvZ;
  float uuvZ = qx * uvY - qy * uvX;

  r.v[0] = x + 2.0f * ( qw * uvX + uuvX );
  r.v[1] = y + 2.0f * ( qw * uvY + uuvY );
  r.v[2] = z + 2.0f * ( qw * uvZ + uuvZ );

  return r;
#endif
}


// Quat fromMat
WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat4)(WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Quat) r;
  const float trace = m.m[0] + m.m[5] + m.m[10];
  if (trace > 0.0) {
    const float root = sqrtf(trace + 1.0f);
    r.v[3] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[0] = (m.m[6] - m.m[9]) * invRoot;
    r.v[1] = (m.m[8] - m.m[2]) * invRoot;
    r.v[2] = (m.m[1] - m.m[4]) * invRoot;
  } else {
    int i = 0;
    if (m.m[5] > m.m[0]) {
      i = 1;
    }
    if (m.m[10] > m.m[i * 4 + i]) {
      i = 2;
    }

    const int j = (i + 1) % 3;
    const int k = (i + 2) % 3;

    const float root =
        sqrtf(m.m[i * 4 + i] - m.m[j * 4 + j] - m.m[k * 4 + k] + 1.0f);
    r.v[i] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[3] = (m.m[j * 4 + k] - m.m[k * 4 + j]) * invRoot;
    r.v[j] = (m.m[j * 4 + i] + m.m[i * 4 + j]) * invRoot;
    r.v[k] = (m.m[k * 4 + i] + m.m[i * 4 + k]) * invRoot;
  }
  return r;
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat3)(WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Quat) r;
  const float trace = m.m[0] + m.m[5] + m.m[10];
  if (trace > 0.0) {
    const float root = sqrtf(trace + 1.0f);
    r.v[3] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[0] = (m.m[6] - m.m[9]) * invRoot;
    r.v[1] = (m.m[8] - m.m[2]) * invRoot;
    r.v[2] = (m.m[1] - m.m[4]) * invRoot;
  } else {
    int i = 0;
    if (m.m[5] > m.m[0]) {
      i = 1;
    }
    if (m.m[10] > m.m[i * 4 + i]) {
      i = 2;
    }

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    const float root =
        sqrtf(m.m[i * 4 + i] - m.m[j * 4 + j] - m.m[k * 4 + k] + 1.0f);
    r.v[i] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[3] = (m.m[j * 4 + k] - m.m[k * 4 + j]) * invRoot;
    r.v[j] = (m.m[j * 4 + i] + m.m[i * 4 + j]) * invRoot;
    r.v[k] = (m.m[k * 4 + i] + m.m[i * 4 + k]) * invRoot;
  }
  return r;
}

// fromEuler
WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_euler)(float x_angle_in_radians, float y_angle_in_radians,
                             float z_angle_in_radians,
                             enum WCN_Math_RotationOrder order) {
  WMATH_TYPE(Quat) r;
  float x_half_angle = x_angle_in_radians * 0.5f;
  float y_half_angle = y_angle_in_radians * 0.5f;
  float z_half_angle = z_angle_in_radians * 0.5f;

  float s_x, c_x, s_y, c_y, s_z, c_z;
#if defined(__GNUC__) && defined(__x86_64__)
  sincosf(x_half_angle, &s_x, &c_x);
  sincosf(y_half_angle, &s_y, &c_y);
  sincosf(z_half_angle, &s_z, &c_z);
#else
  s_x = sinf(x_half_angle);
  c_x = cosf(x_half_angle);
  s_y = sinf(y_half_angle);
  c_y = cosf(y_half_angle);
  s_z = sinf(z_half_angle);
  c_z = cosf(z_half_angle);
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];

    // 构建向量进行并行计算
    __m128 sz_cz_cz_cz = _mm_set_ps(c_z, c_z, c_z, s_z);
    __m128 cx_sx_cx_cx = _mm_set_ps(c_x, c_x, s_x, c_x);
    __m128 cy_cy_sy_cy = _mm_set_ps(c_y, s_y, c_y, c_y);

    __m128 cz_sz_sz_sz = _mm_set_ps(s_z, s_z, s_z, c_z);
    __m128 sx_cx_sx_sx = _mm_set_ps(s_x, s_x, c_x, s_x);
    __m128 sy_sy_cy_sy = _mm_set_ps(s_y, c_y, s_y, s_y);

    // 计算 A 和 B 部分
    __m128 A = _mm_mul_ps(_mm_mul_ps(sz_cz_cz_cz, cx_sx_cx_cx), cy_cy_sy_cy);
    __m128 B = _mm_mul_ps(_mm_mul_ps(cz_sz_sz_sz, sx_cx_sx_sx), sy_sy_cy_sy);

    // 应用符号
    __m128 signs_vec = _mm_set_ps((float)signs[3], (float)signs[2],
                                  (float)signs[1], (float)signs[0]);
    B = _mm_mul_ps(B, signs_vec);

    // 最终结果
    __m128 result = _mm_add_ps(A, B);
    _mm_storeu_ps(r.v, result);
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];

    // 构建向量进行并行计算
    float32x4_t sz_cz_cz_cz = {s_z, c_z, c_z, c_z};
    float32x4_t cx_sx_cx_cx = {c_x, s_x, c_x, c_x};
    float32x4_t cy_cy_sy_cy = {c_y, c_y, s_y, c_y};

    float32x4_t cz_sz_sz_sz = {c_z, s_z, s_z, s_z};
    float32x4_t sx_cx_sx_sx = {s_x, c_x, s_x, s_x};
    float32x4_t sy_sy_cy_sy = {s_y, c_y, s_y, s_y};

    // 计算 A 和 B 部分
    float32x4_t A = vmulq_f32(vmulq_f32(sz_cz_cz_cz, cx_sx_cx_cx), cy_cy_sy_cy);
    float32x4_t B = vmulq_f32(vmulq_f32(cz_sz_sz_sz, sx_cx_sx_sx), sy_sy_cy_sy);

    // 应用符号
    float32x4_t signs_vec = {(float)signs[0], (float)signs[1],
                             (float)signs[2], (float)signs[3]};
    B = vmulq_f32(B, signs_vec);

    // 最终结果
    float32x4_t result = vaddq_f32(A, B);
    vst1q_f32(r.v, result);
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];

    // 构建向量进行并行计算
    v128_t sz_cz_cz_cz = wasm_f32x4_make(s_z, c_z, c_z, c_z);
    v128_t cx_sx_cx_cx = wasm_f32x4_make(c_x, s_x, c_x, c_x);
    v128_t cy_cy_sy_cy = wasm_f32x4_make(c_y, c_y, s_y, c_y);

    v128_t cz_sz_sz_sz = wasm_f32x4_make(c_z, s_z, s_z, s_z);
    v128_t sx_cx_sx_sx = wasm_f32x4_make(s_x, c_x, s_x, s_x);
    v128_t sy_sy_cy_sy = wasm_f32x4_make(s_y, c_y, s_y, s_y);

    // 计算 A 和 B 部分
    v128_t A = wasm_f32x4_mul(wasm_f32x4_mul(sz_cz_cz_cz, cx_sx_cx_cx), cy_cy_sy_cy);
    v128_t B = wasm_f32x4_mul(wasm_f32x4_mul(cz_sz_sz_sz, sx_cx_sx_sx), sy_sy_cy_sy);

    // 应用符号
    v128_t signs_vec = wasm_f32x4_make((float)signs[0], (float)signs[1],
                                      (float)signs[2], (float)signs[3]);
    B = wasm_f32x4_mul(B, signs_vec);

    // 最终结果
    v128_t result = wasm_f32x4_add(A, B);
    wasm_v128_store(r.v, result);
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];
    size_t vl = vsetvlmax_e32m1();

    // 构建向量进行并行计算
    float sz_cz_cz_cz_data[4] = {s_z, c_z, c_z, c_z};
    float cx_sx_cx_cx_data[4] = {c_x, s_x, c_x, c_x};
    float cy_cy_sy_cy_data[4] = {c_y, c_y, s_y, c_y};

    float cz_sz_sz_sz_data[4] = {c_z, s_z, s_z, s_z};
    float sx_cx_sx_sx_data[4] = {s_x, c_x, s_x, s_x};
    float sy_sy_cy_sy_data[4] = {s_y, c_y, s_y, s_y};

    vfloat32m1_t sz_cz_cz_cz = vle32_v_f32m1(sz_cz_cz_cz_data, vl);
    vfloat32m1_t cx_sx_cx_cx = vle32_v_f32m1(cx_sx_cx_cx_data, vl);
    vfloat32m1_t cy_cy_sy_cy = vle32_v_f32m1(cy_cy_sy_cy_data, vl);

    vfloat32m1_t cz_sz_sz_sz = vle32_v_f32m1(cz_sz_sz_sz_data, vl);
    vfloat32m1_t sx_cx_sx_sx = vle32_v_f32m1(sx_cx_sx_sx_data, vl);
    vfloat32m1_t sy_sy_cy_sy = vle32_v_f32m1(sy_sy_cy_sy_data, vl);

    // 计算 A 和 B 部分
    vfloat32m1_t A = vfmul_vv_f32m1(vfmul_vv_f32m1(sz_cz_cz_cz, cx_sx_cx_cx, vl), cy_cy_sy_cy, vl);
    vfloat32m1_t B = vfmul_vv_f32m1(vfmul_vv_f32m1(cz_sz_sz_sz, sx_cx_sx_sx, vl), sy_sy_cy_sy, vl);

    // 应用符号
    float signs_vec_data[4] = {(float)signs[0], (float)signs[1], (float)signs[2], (float)signs[3]};
    vfloat32m1_t signs_vec = vle32_v_f32m1(signs_vec_data, vl);
    B = vfmul_vv_f32m1(B, signs_vec, vl);

    // 最终结果
    vfloat32m1_t result = vfadd_vv_f32m1(A, B, vl);
    vse32_v_f32m1(r.v, result, vl);
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];

    // 构建向量进行并行计算
    __m128 sz_cz_cz_cz = __lsx_vreplfr2vr_s(s_z);
    sz_cz_cz_cz = __lsx_vinsgr2vr_w(sz_cz_cz_cz, __builtin_bit_cast(float, (uint32_t)c_z), 1);
    sz_cz_cz_cz = __lsx_vinsgr2vr_w(sz_cz_cz_cz, __builtin_bit_cast(float, (uint32_t)c_z), 2);
    sz_cz_cz_cz = __lsx_vinsgr2vr_w(sz_cz_cz_cz, __builtin_bit_cast(float, (uint32_t)c_z), 3);

    __m128 cx_sx_cx_cx = __lsx_vreplfr2vr_s(c_x);
    cx_sx_cx_cx = __lsx_vinsgr2vr_w(cx_sx_cx_cx, __builtin_bit_cast(float, (uint32_t)s_x), 1);
    cx_sx_cx_cx = __lsx_vinsgr2vr_w(cx_sx_cx_cx, __builtin_bit_cast(float, (uint32_t)c_x), 2);
    cx_sx_cx_cx = __lsx_vinsgr2vr_w(cx_sx_cx_cx, __builtin_bit_cast(float, (uint32_t)c_x), 3);

    __m128 cy_cy_sy_cy = __lsx_vreplfr2vr_s(c_y);
    cy_cy_sy_cy = __lsx_vinsgr2vr_w(cy_cy_sy_cy, __builtin_bit_cast(float, (uint32_t)c_y), 1);
    cy_cy_sy_cy = __lsx_vinsgr2vr_w(cy_cy_sy_cy, __builtin_bit_cast(float, (uint32_t)s_y), 2);
    cy_cy_sy_cy = __lsx_vinsgr2vr_w(cy_cy_sy_cy, __builtin_bit_cast(float, (uint32_t)c_y), 3);

    __m128 cz_sz_sz_sz = __lsx_vreplfr2vr_s(c_z);
    cz_sz_sz_sz = __lsx_vinsgr2vr_w(cz_sz_sz_sz, __builtin_bit_cast(float, (uint32_t)s_z), 1);
    cz_sz_sz_sz = __lsx_vinsgr2vr_w(cz_sz_sz_sz, __builtin_bit_cast(float, (uint32_t)s_z), 2);
    cz_sz_sz_sz = __lsx_vinsgr2vr_w(cz_sz_sz_sz, __builtin_bit_cast(float, (uint32_t)s_z), 3);

    __m128 sx_cx_sx_sx = __lsx_vreplfr2vr_s(s_x);
    sx_cx_sx_sx = __lsx_vinsgr2vr_w(sx_cx_sx_sx, __builtin_bit_cast(float, (uint32_t)c_x), 1);
    sx_cx_sx_sx = __lsx_vinsgr2vr_w(sx_cx_sx_sx, __builtin_bit_cast(float, (uint32_t)s_x), 2);
    sx_cx_sx_sx = __lsx_vinsgr2vr_w(sx_cx_sx_sx, __builtin_bit_cast(float, (uint32_t)s_x), 3);

    __m128 sy_sy_cy_sy = __lsx_vreplfr2vr_s(s_y);
    sy_sy_cy_sy = __lsx_vinsgr2vr_w(sy_sy_cy_sy, __builtin_bit_cast(float, (uint32_t)c_y), 1);
    sy_sy_cy_sy = __lsx_vinsgr2vr_w(sy_sy_cy_sy, __builtin_bit_cast(float, (uint32_t)s_y), 2);
    sy_sy_cy_sy = __lsx_vinsgr2vr_w(sy_sy_cy_sy, __builtin_bit_cast(float, (uint32_t)s_y), 3);

    // 计算 A 和 B 部分
    __m128 A = __lsx_fmul_s(__lsx_fmul_s(sz_cz_cz_cz, cx_sx_cx_cx), cy_cy_sy_cy);
    __m128 B = __lsx_fmul_s(__lsx_fmul_s(cz_sz_sz_sz, sx_cx_sx_sx), sy_sy_cy_sy);

    // 应用符号
    __m128 signs_vec = __lsx_vreplgr2vr_w(__builtin_bit_cast(float, (uint32_t)signs[0]));
    signs_vec = __lsx_vinsgr2vr_w(signs_vec, __builtin_bit_cast(float, (uint32_t)signs[1]), 1);
    signs_vec = __lsx_vinsgr2vr_w(signs_vec, __builtin_bit_cast(float, (uint32_t)signs[2]), 2);
    signs_vec = __lsx_vinsgr2vr_w(signs_vec, __builtin_bit_cast(float, (uint32_t)signs[3]), 3);
    B = __lsx_fmul_s(B, signs_vec);

    // 最终结果
    __m128 result = __lsx_fadd_s(A, B);
    __lsx_vst(result, r.v, 0);
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#else
  // 标量版本保持不变
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];

    r.v[0] = s_z * c_x * c_y + c_z * s_x * s_y * signs[0];
    r.v[1] = c_z * s_x * c_y + s_z * c_x * s_y * signs[1];
    r.v[2] = c_z * c_x * s_y + s_z * s_x * c_y * signs[2];
    r.v[3] = c_z * c_x * c_y + s_z * s_x * s_y * signs[3];
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#endif

  return r;
}

// BEGIN 3D
// vec3 getTranslation
WMATH_TYPE(Vec3)
WMATH_GET_TRANSLATION(Vec3)(const WMATH_TYPE(Mat4) m) {
  return WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = m.m[12],
      .v_y = m.m[13],
      .v_z = m.m[14],
  });
}

// vec3 getAxis
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_axis)(const WMATH_TYPE(Mat4) m, const int axis) {
  int off = axis * 4;
  return WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = m.m[off + 0],
      .v_y = m.m[off + 1],
      .v_z = m.m[off + 2],
  });
}

// vec3 getScale
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_scale)(WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - parallel computation of three column square sums
  __m128 col_x = _mm_set_ps(0.0f, m.m[2], m.m[1], m.m[0]);
  __m128 col_y = _mm_set_ps(0.0f, m.m[6], m.m[5], m.m[4]);
  __m128 col_z = _mm_set_ps(0.0f, m.m[10], m.m[9], m.m[8]);

  // Square each element
  __m128 sq_x = _mm_mul_ps(col_x, col_x);
  __m128 sq_y = _mm_mul_ps(col_y, col_y);
  __m128 sq_z = _mm_mul_ps(col_z, col_z);

  // Horizontally add to get a sum of squares for each column
  float sum_x = _mm_cvtss_f32(wcn_hadd_ps(sq_x));
  float sum_y = _mm_cvtss_f32(wcn_hadd_ps(sq_y));
  float sum_z = _mm_cvtss_f32(wcn_hadd_ps(sq_z));

  // Take square root
  result.v[0] = sqrtf(sum_x);
  result.v[1] = sqrtf(sum_y);
  result.v[2] = sqrtf(sum_z);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - parallel computation of three column square sums
  float32x4_t col_x = {m.m[0], m.m[1], m.m[2], 0.0f};
  float32x4_t col_y = {m.m[4], m.m[5], m.m[6], 0.0f};
  float32x4_t col_z = {m.m[8], m.m[9], m.m[10], 0.0f};

  // Square each element
  float32x4_t sq_x = vmulq_f32(col_x, col_x);
  float32x4_t sq_y = vmulq_f32(col_y, col_y);
  float32x4_t sq_z = vmulq_f32(col_z, col_z);

  // Horizontally add to get a sum of squares for each column
  float sum_x = wcn_hadd_f32(sq_x);
  float sum_y = wcn_hadd_f32(sq_y);
  float sum_z = wcn_hadd_f32(sq_z);

  // Take square root
  result.v[0] = sqrtf(sum_x);
  result.v[1] = sqrtf(sum_y);
  result.v[2] = sqrtf(sum_z);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - parallel computation of three column square sums
  v128_t col_x = wasm_f32x4_make(m.m[0], m.m[1], m.m[2], 0.0f);
  v128_t col_y = wasm_f32x4_make(m.m[4], m.m[5], m.m[6], 0.0f);
  v128_t col_z = wasm_f32x4_make(m.m[8], m.m[9], m.m[10], 0.0f);

  // Square each element
  v128_t sq_x = wasm_f32x4_mul(col_x, col_x);
  v128_t sq_y = wasm_f32x4_mul(col_y, col_y);
  v128_t sq_z = wasm_f32x4_mul(col_z, col_z);

  // Horizontally add to get a sum of squares for each column
  float sum_x = wcn_hadd_wasm(sq_x);
  float sum_y = wcn_hadd_wasm(sq_y);
  float sum_z = wcn_hadd_wasm(sq_z);

  // Take square root
  result.v[0] = sqrtf(sum_x);
  result.v[1] = sqrtf(sum_y);
  result.v[2] = sqrtf(sum_z);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation - parallel computation of three column square sums
  size_t vl = vsetvlmax_e32m1();

  float col_x_data[4] = {m.m[0], m.m[1], m.m[2], 0.0f};
  float col_y_data[4] = {m.m[4], m.m[5], m.m[6], 0.0f};
  float col_z_data[4] = {m.m[8], m.m[9], m.m[10], 0.0f};

  vfloat32m1_t col_x = vle32_v_f32m1(col_x_data, vl);
  vfloat32m1_t col_y = vle32_v_f32m1(col_y_data, vl);
  vfloat32m1_t col_z = vle32_v_f32m1(col_z_data, vl);

  // Square each element
  vfloat32m1_t sq_x = vfmul_vv_f32m1(col_x, col_x, vl);
  vfloat32m1_t sq_y = vfmul_vv_f32m1(col_y, col_y, vl);
  vfloat32m1_t sq_z = vfmul_vv_f32m1(col_z, col_z, vl);

  // Horizontally add to get a sum of squares for each column
  float sum_x = wcn_hadd_riscv(sq_x, vl);
  float sum_y = wcn_hadd_riscv(sq_y, vl);
  float sum_z = wcn_hadd_riscv(sq_z, vl);

  // Take square root
  result.v[0] = sqrtf(sum_x);
  result.v[1] = sqrtf(sum_y);
  result.v[2] = sqrtf(sum_z);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - parallel computation of three column square sums
  __m128 col_x = __lsx_vreplfr2vr_s(m.m[0]);
  col_x = __lsx_vinsgr2vr_w(col_x, __builtin_bit_cast(float, (uint32_t)m.m[1]), 1);
  col_x = __lsx_vinsgr2vr_w(col_x, __builtin_bit_cast(float, (uint32_t)m.m[2]), 2);
  col_x = __lsx_vinsgr2vr_w(col_x, 0, 3);

  __m128 col_y = __lsx_vreplfr2vr_s(m.m[4]);
  col_y = __lsx_vinsgr2vr_w(col_y, __builtin_bit_cast(float, (uint32_t)m.m[5]), 1);
  col_y = __lsx_vinsgr2vr_w(col_y, __builtin_bit_cast(float, (uint32_t)m.m[6]), 2);
  col_y = __lsx_vinsgr2vr_w(col_y, 0, 3);

  __m128 col_z = __lsx_vreplfr2vr_s(m.m[8]);
  col_z = __lsx_vinsgr2vr_w(col_z, __builtin_bit_cast(float, (uint32_t)m.m[9]), 1);
  col_z = __lsx_vinsgr2vr_w(col_z, __builtin_bit_cast(float, (uint32_t)m.m[10]), 2);
  col_z = __lsx_vinsgr2vr_w(col_z, 0, 3);

  // Square each element
  __m128 sq_x = __lsx_fmul_s(col_x, col_x);
  __m128 sq_y = __lsx_fmul_s(col_y, col_y);
  __m128 sq_z = __lsx_fmul_s(col_z, col_z);

  // Horizontally add to get a sum of squares for each column
  float sum_x = wcn_hadd_loongarch(sq_x);
  float sum_y = wcn_hadd_loongarch(sq_y);
  float sum_z = wcn_hadd_loongarch(sq_z);

  // Take square root
  result.v[0] = sqrtf(sum_x);
  result.v[1] = sqrtf(sum_y);
  result.v[2] = sqrtf(sum_z);

#else
  // Scalar fallback
  float x_x = m.m[0];
  float x_y = m.m[1];
  float x_z = m.m[2];
  float y_x = m.m[4];
  float y_y = m.m[5];
  float y_z = m.m[6];
  float z_x = m.m[8];
  float z_y = m.m[9];
  float z_z = m.m[10];
  result.v[0] = sqrtf(x_x * x_x + x_y * x_y + x_z * x_z);
  result.v[1] = sqrtf(y_x * y_x + y_y * y_y + y_z * y_z);
  result.v[2] = sqrtf(z_x * z_x + z_y * z_y + z_z * z_z);
#endif

  return result;
}

// vec3 rotateX
WMATH_TYPE(Vec3)
WMATH_ROTATE_X(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;

  float s = sinf(rad);
  float c = cosf(rad);

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64

  // --- SSE (fully vectorized, no scalar lane extraction) ---
  // vp = p = a - b  (layout: [px, py, pz, 0])
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 vp = _mm_sub_ps(va, vb);

  // Broadcast py and pz into full registers using shuffle:
  // _MM_SHUFFLE(z,y,x,w) -> builds imm8 = (z<<6)|(y<<4)|(x<<2)|w
  // _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1)) -> [py, py, py, py]
  // _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2)) -> [pz, pz, pz, pz]
  __m128 v_py = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1));
  __m128 v_pz = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2));

  // vector constants
  __m128 v_c = _mm_set1_ps(c);
  __m128 v_s = _mm_set1_ps(s);

  // y' = c*py - s*pz   -> vector: [y', y', y', y']
  __m128 v_y = _mm_sub_ps(_mm_mul_ps(v_py, v_c), _mm_mul_ps(v_pz, v_s));

  // z' = s*py + c*pz  -> vector: [z', z', z', z']
  __m128 v_z = _mm_add_ps(_mm_mul_ps(v_py, v_s), _mm_mul_ps(v_pz, v_c));

  // rx duplicated: [px, px, px, px]
  __m128 v_rx = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(0,0,0,0));

  // Now combine into [rx, y', z', 0] using unpack & shuffle without scalar extracts:
  // v_low  = unpacklo(v_rx, v_y) => [rx, y', rx, y']
  // v_high = unpacklo(v_z, vzero) => [z', 0, z', 0]
  // final = shuffle(v_low, v_high, _MM_SHUFFLE(1,0,1,0)) => [rx, y', z', 0]
  __m128 v_low  = _mm_unpacklo_ps(v_rx, v_y);
  __m128 v_zero = _mm_setzero_ps();
  __m128 v_high = _mm_unpacklo_ps(v_z, v_zero);
  __m128 vres   = _mm_shuffle_ps(v_low, v_high, _MM_SHUFFLE(1,0,1,0));

  // add center back (vb) and store
  vres = _mm_add_ps(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64

  // --- NEON (vectorized; using lane duplication where necessary) ---
  float32x4_t va = wcn_load_vec3_partial(a.v); // [px, py, pz, 0]
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t vp = vsubq_f32(va, vb);

  // Extract lanes (cheap) then duplicate into vectors:
  // Note: vgetq_lane_f32 does a lane read (scalar), then vdupq_n_f32 broadcasts it.
  // This keeps core arithmetic vectorized.
  float px = vgetq_lane_f32(vp, 0);
  float py = vgetq_lane_f32(vp, 1);
  float pz = vgetq_lane_f32(vp, 2);

  float32x4_t v_rx = vdupq_n_f32(px);
  float32x4_t v_py = vdupq_n_f32(py);
  float32x4_t v_pz = vdupq_n_f32(pz);

  float32x4_t v_c = vdupq_n_f32(c);
  float32x4_t v_s = vdupq_n_f32(s);

  float32x4_t v_y = vsubq_f32(vmulq_f32(v_py, v_c), vmulq_f32(v_pz, v_s)); // [y',y',y',y']
  float32x4_t v_z = vaddq_f32(vmulq_f32(v_py, v_s), vmulq_f32(v_pz, v_c)); // [z',z',z',z']

  // Build [rx, y', z', 0]
  // v_low  = [rx, y', rx, y']  via vzip or combine:
  float32x4_t v_low = vcombine_f32(vget_low_f32(v_rx), vget_low_f32(v_y)); // [rx_low0, y_low0, rx_low1, y_low1] -> given duplicates it's [rx,y,rx,y]
  // v_high = [z', 0, z', 0] - construct from v_z and zero
  float32x4_t v_zero = vdupq_n_f32(0.0f);
  float32x4_t v_high = vcombine_f32(vget_low_f32(v_z), vget_low_f32(v_zero)); // [z',0,z',0]

  // shuffle to [rx, y', z', 0] - use vsetq_lane to be explicit & portable
  // Extract needed lanes (they are duplicated so lane 0 is representative)
  float rx_s = vgetq_lane_f32(v_rx, 0);
  float y_s  = vgetq_lane_f32(v_y, 0);
  float z_s  = vgetq_lane_f32(v_z, 0);

  float32x4_t vres = vdupq_n_f32(0.0f);
  vres = vsetq_lane_f32(rx_s, vres, 0);
  vres = vsetq_lane_f32(y_s , vres, 1);
  vres = vsetq_lane_f32(z_s , vres, 2);
  // lane3 stays 0

  // add center and store
  vres = vaddq_f32(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD

  // --- WASM SIMD implementation ---
  v128_t va = wcn_load_vec3_partial_wasm(a.v); // [px, py, pz, 0]
  v128_t vb = wcn_load_vec3_partial_wasm(b.v);
  v128_t vp = wasm_f32x4_sub(va, vb);

  // Extract lanes and broadcast
  float px = wasm_f32x4_extract_lane(vp, 0);
  float py = wasm_f32x4_extract_lane(vp, 1);
  float pz = wasm_f32x4_extract_lane(vp, 2);

  v128_t v_rx = wasm_f32x4_splat(px);
  v128_t v_py = wasm_f32x4_splat(py);
  v128_t v_pz = wasm_f32x4_splat(pz);

  v128_t v_c = wasm_f32x4_splat(c);
  v128_t v_s = wasm_f32x4_splat(s);

  v128_t v_y = wasm_f32x4_sub(
    wasm_f32x4_mul(v_py, v_c),
    wasm_f32x4_mul(v_pz, v_s)
  ); // [y',y',y',y']
  v128_t v_z = wasm_f32x4_add(
    wasm_f32x4_mul(v_py, v_s),
    wasm_f32x4_mul(v_pz, v_c)
  ); // [z',z',z',z']

  // Build [rx, y', z', 0]
  float rx_s = wasm_f32x4_extract_lane(v_rx, 0);
  float y_s  = wasm_f32x4_extract_lane(v_y, 0);
  float z_s  = wasm_f32x4_extract_lane(v_z, 0);

  v128_t vres = wasm_v128_const(
    *(int*)&rx_s, *(int*)&y_s, *(int*)&z_s, 0
  );

  // add center and store
  vres = wasm_f32x4_add(vres, vb);
  wcn_store_vec3_partial_wasm(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR

  // --- RISC-V Vector implementation ---
  size_t vl = vsetvlmax_e32m1();
  vfloat32m1_t va = wcn_load_vec3_partial_riscv(a.v, vl); // [px, py, pz, 0]
  vfloat32m1_t vb = wcn_load_vec3_partial_riscv(b.v, vl);
  vfloat32m1_t vp = vfsub_vv_f32m1(va, vb, vl);

  // Extract values and use them in computation
  float px = vfmv_f_s_f32m1_e32m1(vp, 0);
  float py = vfmv_f_s_f32m1_e32m1(vp, 1);
  float pz = vfmv_f_s_f32m1_e32m1(vp, 2);

  // Compute results
  float y_new = c * py - s * pz;
  float z_new = s * py + c * pz;

  vfloat32m1_t vres = vfmv_v_f_f32m1(0.0f, vl);
  vres = vfmv_s_f32m1_e32m1(vres, px, 0, vl);
  vres = vfmv_s_f32m1_e32m1(vres, y_new, 1, vl);
  vres = vfmv_s_f32m1_e32m1(vres, z_new, 2, vl);

  // add center and store
  vres = vfadd_vv_f32m1(vres, vb, vl);
  wcn_store_vec3_partial_riscv(vec3.v, vres, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX

  // --- LoongArch LSX implementation ---
  __m128 va = wcn_load_vec3_partial_loongarch(a.v); // [px, py, pz, 0]
  __m128 vb = wcn_load_vec3_partial_loongarch(b.v);
  __m128 vp = __lsx_vfsub_s(va, vb);

  // Extract lanes and broadcast
  float px = __lsx_vpickve2gr_w_s(vp, 0);
  float py = __lsx_vpickve2gr_w_s(vp, 1);
  float pz = __lsx_vpickve2gr_w_s(vp, 2);

  __m128 v_rx = __lsx_vreplfr2vr_s(px);
  __m128 v_py = __lsx_vreplfr2vr_s(py);
  __m128 v_pz = __lsx_vreplfr2vr_s(pz);

  __m128 v_c = __lsx_vreplfr2vr_s(c);
  __m128 v_s = __lsx_vreplfr2vr_s(s);

  __m128 v_y = __lsx_vfsub_s(__lsx_vfmul_s(v_py, v_c), __lsx_vfmul_s(v_pz, v_s)); // [y',y',y',y']
  __m128 v_z = __lsx_vfadd_s(__lsx_vfmul_s(v_py, v_s), __lsx_vfmul_s(v_pz, v_c)); // [z',z',z',z']

  // Build [rx, y', z', 0]
  __m128 vres = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&(float){0.0f}, 0), px, 0), y_s, 1), z_s, 2);
  float y_s = __lsx_vpickve2gr_w_s(v_y, 0);
  float z_s = __lsx_vpickve2gr_w_s(v_z, 0);

  vres = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&(float){0.0f}, 0), px, 0), y_s, 1), z_s, 2);

  // add center and store
  vres = __lsx_vfadd_s(vres, vb);
  wcn_store_vec3_partial_loongarch(vec3.v, vres);

#else

  // --- Scalar fallback (reference) ---
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = p.v[0];
  r.v[1] = cosf(rad) * p.v[1] - sinf(rad) * p.v[2];
  r.v[2] = sinf(rad) * p.v[1] + cosf(rad) * p.v[2];
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];

#endif

  return vec3;
}

// vec3 rotateY
WMATH_TYPE(Vec3)
WMATH_ROTATE_Y(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;

  float s = sinf(rad);
  float c = cosf(rad);

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64

  // --- SSE (fully vectorized) ---
  __m128 va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 vp = _mm_sub_ps(va, vb);         // p = a - b

  // Broadcast lanes:
  __m128 v_px = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(0,0,0,0)); // [px,px,px,px]
  __m128 v_py = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1)); // [py,py,py,py]
  __m128 v_pz = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2)); // [pz,pz,pz,pz]

  __m128 v_c  = _mm_set1_ps(c);
  __m128 v_s  = _mm_set1_ps(s);
  __m128 v_ns = _mm_set1_ps(-s);
  __m128 v_zero = _mm_setzero_ps();

  // x' = x*c + z*s
  __m128 v_x = _mm_add_ps(_mm_mul_ps(v_px, v_c), _mm_mul_ps(v_pz, v_s));

  // z' = -x*s + z*c
  __m128 v_z = _mm_add_ps(_mm_mul_ps(v_px, v_ns), _mm_mul_ps(v_pz, v_c));

  // Combine into [x', y, z', 0]
  // v_low = [x', y, x', y]
  __m128 v_low  = _mm_unpacklo_ps(v_x, v_py);
  // v_high = [z', 0, z', 0]
  __m128 v_high = _mm_unpacklo_ps(v_z, v_zero);
  // final: [x', y, z', 0]
  __m128 vres = _mm_shuffle_ps(v_low, v_high, _MM_SHUFFLE(1,0,1,0));

  // add center and store
  vres = _mm_add_ps(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64

  // --- NEON (vectorized; using lane-read + broadcast) ---
  float32x4_t va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t vp = vsubq_f32(va, vb);

  // read lanes to scalars then broadcast (common & efficient)
  float px = vgetq_lane_f32(vp, 0);
  float py = vgetq_lane_f32(vp, 1);
  float pz = vgetq_lane_f32(vp, 2);

  float32x4_t v_px = vdupq_n_f32(px);
  float32x4_t v_py = vdupq_n_f32(py);
  float32x4_t v_pz = vdupq_n_f32(pz);

  float32x4_t v_c  = vdupq_n_f32(c);
  float32x4_t v_s  = vdupq_n_f32(s);
  float32x4_t v_ns = vdupq_n_f32(-s);

  // x' = x*c + z*s
  float32x4_t v_x = vaddq_f32(vmulq_f32(v_px, v_c), vmulq_f32(v_pz, v_s));
  // z' = -x*s + z*c
  float32x4_t v_z = vaddq_f32(vmulq_f32(v_px, v_ns), vmulq_f32(v_pz, v_c));

  // Build result [x', y, z', 0]
  float32x4_t vres = vdupq_n_f32(0.0f);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_x, 0), vres, 0);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_py, 0), vres, 1);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_z, 0), vres, 2);
  // lane3 remains 0

  // add back center and store
  vres = vaddq_f32(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD

  // --- WASM SIMD implementation ---
  v128_t va = wcn_load_vec3_partial_wasm(a.v); // [px, py, pz, 0]
  v128_t vb = wcn_load_vec3_partial_wasm(b.v);
  v128_t vp = wasm_f32x4_sub(va, vb);

  // Extract lanes and broadcast
  float px = wasm_f32x4_extract_lane(vp, 0);
  float py = wasm_f32x4_extract_lane(vp, 1);
  float pz = wasm_f32x4_extract_lane(vp, 2);

  v128_t v_px = wasm_f32x4_splat(px);
  v128_t v_py = wasm_f32x4_splat(py);
  v128_t v_pz = wasm_f32x4_splat(pz);

  v128_t v_c = wasm_f32x4_splat(c);
  v128_t v_s = wasm_f32x4_splat(s);
  v128_t v_ns = wasm_f32x4_splat(-s);

  // x' = x*c + z*s
  v128_t v_x = wasm_f32x4_add(
    wasm_f32x4_mul(v_px, v_c),
    wasm_f32x4_mul(v_pz, v_s)
  ); // [x',x',x',x']

  // z' = -x*s + z*c
  v128_t v_z = wasm_f32x4_add(
    wasm_f32x4_mul(v_px, v_ns),
    wasm_f32x4_mul(v_pz, v_c)
  ); // [z',z',z',z']

  // Build [x', y, z', 0]
  float x_s = wasm_f32x4_extract_lane(v_x, 0);
  float z_s = wasm_f32x4_extract_lane(v_z, 0);

  v128_t vres = wasm_v128_const(
    *(int*)&x_s, *(int*)&py, *(int*)&z_s, 0
  );

  // add center and store
  vres = wasm_f32x4_add(vres, vb);
  wcn_store_vec3_partial_wasm(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR

  // --- RISC-V Vector implementation ---
  size_t vl = vsetvlmax_e32m1();
  vfloat32m1_t va = wcn_load_vec3_partial_riscv(a.v, vl); // [px, py, pz, 0]
  vfloat32m1_t vb = wcn_load_vec3_partial_riscv(b.v, vl);
  vfloat32m1_t vp = vfsub_vv_f32m1(va, vb, vl);

  // Extract values and use them in computation
  float px = vfmv_f_s_f32m1_e32m1(vp, 0);
  float py = vfmv_f_s_f32m1_e32m1(vp, 1);
  float pz = vfmv_f_s_f32m1_e32m1(vp, 2);

  // Compute results
  float x_new = c * px + s * pz;
  float z_new = -s * px + c * pz;

  vfloat32m1_t vres = vfmv_v_f_f32m1(0.0f, vl);
  vres = vfmv_s_f32m1_e32m1(vres, x_new, 0, vl);
  vres = vfmv_s_f32m1_e32m1(vres, py, 1, vl);
  vres = vfmv_s_f32m1_e32m1(vres, z_new, 2, vl);

  // add center and store
  vres = vfadd_vv_f32m1(vres, vb, vl);
  wcn_store_vec3_partial_riscv(vec3.v, vres, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX

  // --- LoongArch LSX implementation ---
  __m128 va = wcn_load_vec3_partial_loongarch(a.v); // [px, py, pz, 0]
  __m128 vb = wcn_load_vec3_partial_loongarch(b.v);
  __m128 vp = __lsx_vfsub_s(va, vb);

  // Extract lanes and broadcast
  float px = __lsx_vpickve2gr_w_s(vp, 0);
  float py = __lsx_vpickve2gr_w_s(vp, 1);
  float pz = __lsx_vpickve2gr_w_s(vp, 2);

  __m128 v_px = __lsx_vreplfr2vr_s(px);
  __m128 v_py = __lsx_vreplfr2vr_s(py);
  __m128 v_pz = __lsx_vreplfr2vr_s(pz);

  __m128 v_c = __lsx_vreplfr2vr_s(c);
  __m128 v_s = __lsx_vreplfr2vr_s(s);
  __m128 v_ns = __lsx_vreplfr2vr_s(-s);

  // x' = x*c + z*s
  __m128 v_x = __lsx_vfadd_s(__lsx_vfmul_s(v_px, v_c), __lsx_vfmul_s(v_pz, v_s)); // [x',x',x',x']

  // z' = -x*s + z*c
  __m128 v_z = __lsx_vfadd_s(__lsx_vfmul_s(v_px, v_ns), __lsx_vfmul_s(v_pz, v_c)); // [z',z',z',z']

  // Build [x', y, z', 0]
  float x_s = __lsx_vpickve2gr_w_s(v_x, 0);
  float z_s = __lsx_vpickve2gr_w_s(v_z, 0);

  __m128 vres = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&(float){0.0f}, 0), x_s, 0), py, 1), z_s, 2);

  // add center and store
  vres = __lsx_vfadd_s(vres, vb);
  wcn_store_vec3_partial_loongarch(vec3.v, vres);

#else

  // --- Scalar fallback (reference) ---
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = sinf(rad) * p.v[2] + cosf(rad) * p.v[0]; // x' = x*c + z*s  (note: sin* z + cos*x same as written)
  r.v[1] = p.v[1];
  r.v[2] = cosf(rad) * p.v[2] - sinf(rad) * p.v[0]; // z' = -x*s + z*c
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];

#endif

  return vec3;
}

// vec3 rotateZ
WMATH_TYPE(Vec3)
WMATH_ROTATE_Z(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;

  float s = sinf(rad);
  float c = cosf(rad);

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64

  // --- SSE (fully vectorized) ---
  __m128 va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 vp = _mm_sub_ps(va, vb);         // p = a - b

  // Broadcast lanes
  __m128 v_px = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(0,0,0,0)); // [px,px,px,px]
  __m128 v_py = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1)); // [py,py,py,py]
  __m128 v_pz = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2)); // [pz,pz,pz,pz]

  __m128 v_c = _mm_set1_ps(c);
  __m128 v_s = _mm_set1_ps(s);
  __m128 v_zero = _mm_setzero_ps();

  // x' = x*c - y*s
  __m128 v_x = _mm_sub_ps(_mm_mul_ps(v_px, v_c), _mm_mul_ps(v_py, v_s));

  // y' = x*s + y*c
  __m128 v_y = _mm_add_ps(_mm_mul_ps(v_px, v_s), _mm_mul_ps(v_py, v_c));

  // z' = z (unchanged by Z-rotation)
  __m128 v_z = v_pz;

  // Combine into [x', y', z', 0]
  // v_low  = [x', y', x', y']
  __m128 v_low  = _mm_unpacklo_ps(v_x, v_y);
  // v_high = [z', 0, z', 0]
  __m128 v_high = _mm_unpacklo_ps(v_z, v_zero);
  // final = [x', y', z', 0]
  __m128 vres = _mm_shuffle_ps(v_low, v_high, _MM_SHUFFLE(1,0,1,0));

  // add center back and store
  vres = _mm_add_ps(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64

  // --- NEON (vectorized; lane-read + broadcast) ---
  float32x4_t va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t vp = vsubq_f32(va, vb);

  // read lanes then broadcast
  float px = vgetq_lane_f32(vp, 0);
  float py = vgetq_lane_f32(vp, 1);
  float pz = vgetq_lane_f32(vp, 2);

  float32x4_t v_px = vdupq_n_f32(px);
  float32x4_t v_py = vdupq_n_f32(py);
  float32x4_t v_pz = vdupq_n_f32(pz);

  float32x4_t v_c = vdupq_n_f32(c);
  float32x4_t v_s = vdupq_n_f32(s);

  // x' = x*c - y*s
  float32x4_t v_x = vsubq_f32(vmulq_f32(v_px, v_c), vmulq_f32(v_py, v_s));
  // y' = x*s + y*c
  float32x4_t v_y = vaddq_f32(vmulq_f32(v_px, v_s), vmulq_f32(v_py, v_c));
  // z' = z
  float32x4_t v_z = v_pz;

  // Build [x', y', z', 0] explicitly
  float32x4_t vres = vdupq_n_f32(0.0f);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_x, 0), vres, 0);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_y, 0), vres, 1);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_z, 0), vres, 2);
  // lane3 stays 0

  // add center and store
  vres = vaddq_f32(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD

  // --- WASM SIMD implementation ---
  v128_t va = wcn_load_vec3_partial_wasm(a.v); // [px, py, pz, 0]
  v128_t vb = wcn_load_vec3_partial_wasm(b.v);
  v128_t vp = wasm_f32x4_sub(va, vb);

  // Extract lanes and broadcast
  float px = wasm_f32x4_extract_lane(vp, 0);
  float py = wasm_f32x4_extract_lane(vp, 1);
  float pz = wasm_f32x4_extract_lane(vp, 2);

  v128_t v_px = wasm_f32x4_splat(px);
  v128_t v_py = wasm_f32x4_splat(py);
  v128_t v_pz = wasm_f32x4_splat(pz);

  v128_t v_c = wasm_f32x4_splat(c);
  v128_t v_s = wasm_f32x4_splat(s);

  // x' = x*c - y*s
  v128_t v_x = wasm_f32x4_sub(
    wasm_f32x4_mul(v_px, v_c),
    wasm_f32x4_mul(v_py, v_s)
  ); // [x',x',x',x']

  // y' = x*s + y*c
  v128_t v_y = wasm_f32x4_add(
    wasm_f32x4_mul(v_px, v_s),
    wasm_f32x4_mul(v_py, v_c)
  ); // [y',y',y',y']

  // z' = z (unchanged)
  v128_t v_z = v_pz;

  // Build [x', y', z', 0]
  float x_s = wasm_f32x4_extract_lane(v_x, 0);
  float y_s = wasm_f32x4_extract_lane(v_y, 0);

  v128_t vres = wasm_v128_const(
    *(int*)&x_s, *(int*)&y_s, *(int*)&pz, 0
  );

  // add center and store
  vres = wasm_f32x4_add(vres, vb);
  wcn_store_vec3_partial_wasm(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR

  // --- RISC-V Vector implementation ---
  size_t vl = vsetvlmax_e32m1();
  vfloat32m1_t va = wcn_load_vec3_partial_riscv(a.v, vl); // [px, py, pz, 0]
  vfloat32m1_t vb = wcn_load_vec3_partial_riscv(b.v, vl);
  vfloat32m1_t vp = vfsub_vv_f32m1(va, vb, vl);

  // Extract values and use them in computation
  float px = vfmv_f_s_f32m1_e32m1(vp, 0);
  float py = vfmv_f_s_f32m1_e32m1(vp, 1);
  float pz = vfmv_f_s_f32m1_e32m1(vp, 2);

  // Compute results
  float x_new = c * px - s * py;
  float y_new = s * px + c * py;

  vfloat32m1_t vres = vfmv_v_f_f32m1(0.0f, vl);
  vres = vfmv_s_f32m1_e32m1(vres, x_new, 0, vl);
  vres = vfmv_s_f32m1_e32m1(vres, y_new, 1, vl);
  vres = vfmv_s_f32m1_e32m1(vres, pz, 2, vl);

  // add center and store
  vres = vfadd_vv_f32m1(vres, vb, vl);
  wcn_store_vec3_partial_riscv(vec3.v, vres, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX

  // --- LoongArch LSX implementation ---
  __m128 va = wcn_load_vec3_partial_loongarch(a.v); // [px, py, pz, 0]
  __m128 vb = wcn_load_vec3_partial_loongarch(b.v);
  __m128 vp = __lsx_vfsub_s(va, vb);

  // Extract lanes and broadcast
  float px = __lsx_vpickve2gr_w_s(vp, 0);
  float py = __lsx_vpickve2gr_w_s(vp, 1);
  float pz = __lsx_vpickve2gr_w_s(vp, 2);

  __m128 v_px = __lsx_vreplfr2vr_s(px);
  __m128 v_py = __lsx_vreplfr2vr_s(py);
  __m128 v_pz = __lsx_vreplfr2vr_s(pz);

  __m128 v_c = __lsx_vreplfr2vr_s(c);
  __m128 v_s = __lsx_vreplfr2vr_s(s);

  // x' = x*c - y*s
  __m128 v_x = __lsx_vfsub_s(__lsx_vfmul_s(v_px, v_c), __lsx_vfmul_s(v_py, v_s)); // [x',x',x',x']

  // y' = x*s + y*c
  __m128 v_y = __lsx_vfadd_s(__lsx_vfmul_s(v_px, v_s), __lsx_vfmul_s(v_py, v_c)); // [y',y',y',y']

  // z' = z (unchanged)
  __m128 v_z = v_pz;

  // Build [x', y', z', 0]
  float x_s = __lsx_vpickve2gr_w_s(v_x, 0);
  float y_s = __lsx_vpickve2gr_w_s(v_y, 0);

  __m128 vres = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&(float){0.0f}, 0), x_s, 0), y_s, 1), pz, 2);

  // add center and store
  vres = __lsx_vfadd_s(vres, vb);
  wcn_store_vec3_partial_loongarch(vec3.v, vres);

#else

  // --- Scalar fallback (reference) ---
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = cosf(rad) * p.v[0] - sinf(rad) * p.v[1]; // x' = x*c - y*s
  r.v[1] = sinf(rad) * p.v[0] + cosf(rad) * p.v[1]; // y' = x*s + y*c
  r.v[2] = p.v[2];                                  // z unchanged
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];

#endif

  return vec3;
}

// vec4 transformMat4
WMATH_TYPE(Vec4)
WMATH_CALL(Vec4, transform_mat4)(WMATH_TYPE(Vec4) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 v_vec = _mm_loadu_ps(v.v); // [x, y, z, w]
  
  // Load Columns
  __m128 col0 = _mm_loadu_ps(&m.m[0]);
  __m128 col1 = _mm_loadu_ps(&m.m[4]);
  __m128 col2 = _mm_loadu_ps(&m.m[8]);
  __m128 col3 = _mm_loadu_ps(&m.m[12]);

  // Broadcast components
  __m128 v_x = _mm_shuffle_ps(v_vec, v_vec, 0x00);
  __m128 v_y = _mm_shuffle_ps(v_vec, v_vec, 0x55);
  __m128 v_z = _mm_shuffle_ps(v_vec, v_vec, 0xAA);
  __m128 v_w = _mm_shuffle_ps(v_vec, v_vec, 0xFF);

  // Accumulate
#if defined(WCN_HAS_FMA)
  __m128 res = _mm_mul_ps(col0, v_x);
  res = _mm_fmadd_ps(col1, v_y, res);
  res = _mm_fmadd_ps(col2, v_z, res);
  res = _mm_fmadd_ps(col3, v_w, res);
#else
  __m128 res = _mm_mul_ps(col0, v_x);
  res = _mm_add_ps(res, _mm_mul_ps(col1, v_y));
  res = _mm_add_ps(res, _mm_mul_ps(col2, v_z));
  res = _mm_add_ps(res, _mm_mul_ps(col3, v_w));
#endif

  _mm_storeu_ps(result.v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // ================= NEON 实现 =================
  float32x4_t vec_v = vld1q_f32(v.v);

  // 按行加载矩阵
  float32x4_t row0 = {m.m[0],  m.m[4],  m.m[8],  m.m[12]};
  float32x4_t row1 = {m.m[1],  m.m[5],  m.m[9],  m.m[13]};
  float32x4_t row2 = {m.m[2],  m.m[6],  m.m[10], m.m[14]};
  float32x4_t row3 = {m.m[3],  m.m[7],  m.m[11], m.m[15]};

  // 点积
  float32x4_t mul0 = vmulq_f32(row0, vec_v);
  float32x4_t mul1 = vmulq_f32(row1, vec_v);
  float32x4_t mul2 = vmulq_f32(row2, vec_v);
  float32x4_t mul3 = vmulq_f32(row3, vec_v);

  float x = vaddvq_f32(mul0); // 横向加
  float y = vaddvq_f32(mul1);
  float z = vaddvq_f32(mul2);
  float w = vaddvq_f32(mul3);

  float32x4_t result_v = {x, y, z, w};
  vst1q_f32(result.v, result_v);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // ================= WASM SIMD 实现 =================
  v128_t vec_v = wasm_v128_load(v.v);

  // 按行加载矩阵
  v128_t row0 = wasm_f32x4_make(m.m[0],  m.m[4],  m.m[8],  m.m[12]);
  v128_t row1 = wasm_f32x4_make(m.m[1],  m.m[5],  m.m[9],  m.m[13]);
  v128_t row2 = wasm_f32x4_make(m.m[2],  m.m[6],  m.m[10], m.m[14]);
  v128_t row3 = wasm_f32x4_make(m.m[3],  m.m[7],  m.m[11], m.m[15]);

  // 点积
  v128_t mul0 = wasm_f32x4_mul(row0, vec_v);
  v128_t mul1 = wasm_f32x4_mul(row1, vec_v);
  v128_t mul2 = wasm_f32x4_mul(row2, vec_v);
  v128_t mul3 = wasm_f32x4_mul(row3, vec_v);

  // 横向加法 (使用wasm_f32x4_extract_lane提取特定lane然后相加)
  float x = wasm_f32x4_extract_lane(mul0, 0) + wasm_f32x4_extract_lane(mul0, 1) +
            wasm_f32x4_extract_lane(mul0, 2) + wasm_f32x4_extract_lane(mul0, 3);
  float y = wasm_f32x4_extract_lane(mul1, 0) + wasm_f32x4_extract_lane(mul1, 1) +
            wasm_f32x4_extract_lane(mul1, 2) + wasm_f32x4_extract_lane(mul1, 3);
  float z = wasm_f32x4_extract_lane(mul2, 0) + wasm_f32x4_extract_lane(mul2, 1) +
            wasm_f32x4_extract_lane(mul2, 2) + wasm_f32x4_extract_lane(mul2, 3);
  float w = wasm_f32x4_extract_lane(mul3, 0) + wasm_f32x4_extract_lane(mul3, 1) +
            wasm_f32x4_extract_lane(mul3, 2) + wasm_f32x4_extract_lane(mul3, 3);

  result.v[0] = x;
  result.v[1] = y;
  result.v[2] = z;
  result.v[3] = w;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // ================= RISC-V Vector 实现 =================
  size_t vl = __riscv_vsetvli(4, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);

  // 加载向量
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, vl);

  // 按行加载矩阵
  vfloat32m1_t row0 = __riscv_vcreate_v_f32m1x4(m.m[0],  m.m[4],  m.m[8],  m.m[12]);
  vfloat32m1_t row1 = __riscv_vcreate_v_f32m1x4(m.m[1],  m.m[5],  m.m[9],  m.m[13]);
  vfloat32m1_t row2 = __riscv_vcreate_v_f32m1x4(m.m[2],  m.m[6],  m.m[10], m.m[14]);
  vfloat32m1_t row3 = __riscv_vcreate_v_f32m1x4(m.m[3],  m.m[7],  m.m[11], m.m[15]);

  // 点积
  vfloat32m1_t mul0 = __riscv_vfmul_vv_f32m1(row0, vec_v, vl);
  vfloat32m1_t mul1 = __riscv_vfmul_vv_f32m1(row1, vec_v, vl);
  vfloat32m1_t mul2 = __riscv_vfmul_vv_f32m1(row2, vec_v, vl);
  vfloat32m1_t mul3 = __riscv_vfmul_vv_f32m1(row3, vec_v, vl);

  // 横向加法
  float x = __riscv_vfmv_f_s_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(mul0, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl), vl);
  float y = __riscv_vfmv_f_s_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(mul1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl), vl);
  float z = __riscv_vfmv_f_s_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(mul2, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl), vl);
  float w = __riscv_vfmv_f_s_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(mul3, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl), vl);

  result.v[0] = x;
  result.v[1] = y;
  result.v[2] = z;
  result.v[3] = w;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // ================= LoongArch LSX 实现 =================
  __m128 vec_v = __lsx_vld(v.v, 0);

  // 按行加载矩阵
  __m128 row0 = __lsx_vldrepl_w((float*)&m.m[0], 0);
  row0 = __lsx_vinsgr2vr_w(row0, m.m[4], 1);
  row0 = __lsx_vinsgr2vr_w(row0, m.m[8], 2);
  row0 = __lsx_vinsgr2vr_w(row0, m.m[12], 3);

  __m128 row1 = __lsx_vldrepl_w((float*)&m.m[1], 0);
  row1 = __lsx_vinsgr2vr_w(row1, m.m[5], 1);
  row1 = __lsx_vinsgr2vr_w(row1, m.m[9], 2);
  row1 = __lsx_vinsgr2vr_w(row1, m.m[13], 3);

  __m128 row2 = __lsx_vldrepl_w((float*)&m.m[2], 0);
  row2 = __lsx_vinsgr2vr_w(row2, m.m[6], 1);
  row2 = __lsx_vinsgr2vr_w(row2, m.m[10], 2);
  row2 = __lsx_vinsgr2vr_w(row2, m.m[14], 3);

  __m128 row3 = __lsx_vldrepl_w((float*)&m.m[3], 0);
  row3 = __lsx_vinsgr2vr_w(row3, m.m[7], 1);
  row3 = __lsx_vinsgr2vr_w(row3, m.m[11], 2);
  row3 = __lsx_vinsgr2vr_w(row3, m.m[15], 3);

  // 点积
  __m128 mul0 = __lsx_vfmul_s(row0, vec_v);
  __m128 mul1 = __lsx_vfmul_s(row1, vec_v);
  __m128 mul2 = __lsx_vfmul_s(row2, vec_v);
  __m128 mul3 = __lsx_vfmul_s(row3, vec_v);

  // 横向加法 (需要手动实现)
  float x = __lsx_vpickve2gr_w(mul0, 0) + __lsx_vpickve2gr_w(mul0, 1) +
            __lsx_vpickve2gr_w(mul0, 2) + __lsx_vpickve2gr_w(mul0, 3);
  float y = __lsx_vpickve2gr_w(mul1, 0) + __lsx_vpickve2gr_w(mul1, 1) +
            __lsx_vpickve2gr_w(mul1, 2) + __lsx_vpickve2gr_w(mul1, 3);
  float z = __lsx_vpickve2gr_w(mul2, 0) + __lsx_vpickve2gr_w(mul2, 1) +
            __lsx_vpickve2gr_w(mul2, 2) + __lsx_vpickve2gr_w(mul2, 3);
  float w = __lsx_vpickve2gr_w(mul3, 0) + __lsx_vpickve2gr_w(mul3, 1) +
            __lsx_vpickve2gr_w(mul3, 2) + __lsx_vpickve2gr_w(mul3, 3);

  result.v[0] = x;
  result.v[1] = y;
  result.v[2] = z;
  result.v[3] = w;

#else
  // ================= 标量回退 =================
  const float x = v.v[0];
  const float y = v.v[1];
  const float z = v.v[2];
  const float w = v.v[3];

  result.v[0] = m.m[0] * x + m.m[4] * y + m.m[8]  * z + m.m[12] * w;
  result.v[1] = m.m[1] * x + m.m[5] * y + m.m[9]  * z + m.m[13] * w;
  result.v[2] = m.m[2] * x + m.m[6] * y + m.m[10] * z + m.m[14] * w;
  result.v[3] = m.m[3] * x + m.m[7] * y + m.m[11] * z + m.m[15] * w;
#endif

  return result;
}

// Quat rotate_x
WMATH_TYPE(Quat)
WMATH_ROTATE_X(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  __m128 q_vec = _mm_loadu_ps(q.v);

  // Create rotation quaternion b = [s, 0, 0, c]
  __m128 b_vec = _mm_set_ps(c, 0.0f, 0.0f, s);

  // Shuffle q for multiplication:
  // q = [x, y, z, w]
  // For result.x: q.x * b.w + q.w * b.x = q[0] * b[3] + q[3] * b[0]
  // For result.y: q.y * b.w + q.z * b.x = q[1] * b[3] + q[2] * b[0]
  // For result.z: q.z * b.w - q.y * b.x = q[2] * b[3] - q[1] * b[0]
  // For result.w: q.w * b.w - q.x * b.x = q[3] * b[3] - q[0] * b[0]

  const __m128 q_swapped = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(0, 3, 2, 1)); // [y, z, w, x]
  const __m128 b_swapped = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(0, 0, 0, 3)); // [c, 0, 0, c]

  __m128 mul1 = _mm_mul_ps(q_vec, b_swapped);
  const __m128 mul2 = _mm_mul_ps(q_swapped, _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3, 0, 0, 0))); // [s, 0, 0, s]

  // Add/sub according to formula
  const __m128 signs = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
  const __m128 mul2_signed = _mm_mul_ps(mul2, signs);
  __m128 res_vec = _mm_add_ps(mul1, mul2_signed);

  // Correct order: [x, y, z, w]
  res_vec = _mm_shuffle_ps(res_vec, res_vec, _MM_SHUFFLE(2, 1, 0, 3)); // [w, z, y, x] -> [x, y, z, w]
  _mm_storeu_ps(result.v, res_vec);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  float32x4_t q_vec = vld1q_f32(q.v);

  // Create rotation quaternion b = [s, 0, 0, c]
  float32x4_t b_vec = {s, 0.0f, 0.0f, c};

  // Perform quaternion multiplication q * b for rotation around x-axis
  // result.x = q.x * b.w + q.w * b.x = q[0] * b[3] + q[3] * b[0]
  // result.y = q.y * b.w + q.z * b.x = q[1] * b[3] + q[2] * b[0]
  // result.z = q.z * b.w - q.y * b.x = q[2] * b[3] - q[1] * b[0]
  // result.w = q.w * b.w - q.x * b.x = q[3] * b[3] - q[0] * b[0]

  float32x4_t q_components = q_vec;
  float32x2_t qw_qx = vget_low_f32(q_components);     // [q.x, q.y]
  float32x2_t qz_qw = vget_high_f32(q_components);    // [q.z, q.w]

  // Create [b.w, b.x, b.w, b.x] = [c, s, c, s]
  float32x4_t bw_bx = vsetq_lane_f32(c, vdupq_n_f32(s), 0);
  bw_bx = vsetq_lane_f32(c, bw_bx, 2);

  // Create [q.x, q.z, q.w, q.y]
  float32x4_t q_xzw_y = {vgetq_lane_f32(q_vec, 0), vgetq_lane_f32(q_vec, 2),
                         vgetq_lane_f32(q_vec, 3), vgetq_lane_f32(q_vec, 1)};

  // Create [b.w, b.x, b.w, b.x]
  float32x4_t b_wx_wx = {c, s, c, s};

  // Multiply
  float32x4_t mul1 = vmulq_f32(q_vec, vsetq_lane_f32(c, vsetq_lane_f32(c, vsetq_lane_f32(c, vdupq_n_f32(c), 3), 2), 0));
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 0), mul1, 0);
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 1), mul1, 1);
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 2), mul1, 2);
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 3), mul1, 3);

  // Simpler approach for NEON:
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c + q_w * s;
  result.v[1] = q_y * c + q_z * s;
  result.v[2] = q_z * c - q_y * s;
  result.v[3] = q_w * c - q_x * s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  v128_t q_vec = wasm_v128_load(q.v);

  // Create rotation quaternion b = [s, 0, 0, c]
  v128_t b_vec = wasm_f32x4_make(s, 0.0f, 0.0f, c);

  // Perform quaternion multiplication manually
  float q_x = wasm_f32x4_extract_lane(q_vec, 0);
  float q_y = wasm_f32x4_extract_lane(q_vec, 1);
  float q_z = wasm_f32x4_extract_lane(q_vec, 2);
  float q_w = wasm_f32x4_extract_lane(q_vec, 3);

  result.v[0] = q_x * c + q_w * s;
  result.v[1] = q_y * c + q_z * s;
  result.v[2] = q_z * c - q_y * s;
  result.v[3] = q_w * c - q_x * s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  size_t vl = __riscv_vsetvli(4, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);

  // Load quaternion
  vfloat32m1_t q_vec = __riscv_vle32_v_f32m1(q.v, vl);

  // Create rotation quaternion b = [s, 0, 0, c]
  vfloat32m1_t b_vec = __riscv_vcreate_v_f32m1x4(s, 0.0f, 0.0f, c);

  // Perform quaternion multiplication manually
  float q_x = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(q_vec, 0), vl);
  float q_y = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(q_vec, 1), vl);
  float q_z = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(q_vec, 2), vl);
  float q_w = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(q_vec, 3), vl);

  result.v[0] = q_x * c + q_w * s;
  result.v[1] = q_y * c + q_z * s;
  result.v[2] = q_z * c - q_y * s;
  result.v[3] = q_w * c - q_x * s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  __m128 q_vec = __lsx_vld(q.v, 0);

  // Create rotation quaternion b = [s, 0, 0, c]
  __m128 b_vec = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&s, 0), 0.0f, 1), 0.0f, 2), c, 3);

  // Perform quaternion multiplication manually
  float q_x = __lsx_vpickve2gr_w(q_vec, 0);
  float q_y = __lsx_vpickve2gr_w(q_vec, 1);
  float q_z = __lsx_vpickve2gr_w(q_vec, 2);
  float q_w = __lsx_vpickve2gr_w(q_vec, 3);

  result.v[0] = q_x * c + q_w * s;
  result.v[1] = q_y * c + q_z * s;
  result.v[2] = q_z * c - q_y * s;
  result.v[3] = q_w * c - q_x * s;

#else
  // Scalar implementation
  float half_angle = angleInRadians * 0.5f;
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  float b_x = sinf(half_angle);
  float b_w = cosf(half_angle);
  result.v[0] = q_x * b_w + q_w * b_x;
  result.v[1] = q_y * b_w + q_z * b_x;
  result.v[2] = q_z * b_w - q_y * b_x;
  result.v[3] = q_w * b_w - q_x * b_x;
#endif

  return result;
}

// Quat rotate_y
WMATH_TYPE(Quat)
WMATH_ROTATE_Y(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  __m128 q_vec = _mm_loadu_ps(q.v);

  // Create rotation quaternion b = [0, s, 0, c]
  _mm_set_ps(c, 0.0f, s, 0.0f);

  // For rotation around y-axis:
  // result.x = q.x * b.w - q.z * b.y = q[0] * b[3] - q[2] * b[1]
  // result.y = q.y * b.w + q.w * b.y = q[1] * b[3] + q[3] * b[1]
  // result.z = q.z * b.w + q.x * b.y = q[2] * b[3] + q[0] * b[1]
  // result.w = q.w * b.w - q.y * b.y = q[3] * b[3] - q[1] * b[1]

  const __m128 mul1 = _mm_mul_ps(q_vec, _mm_set_ps(c, c, c, c));
  const __m128 q_shuffled = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(1, 0, 3, 2)); // [z, w, x, y]
  const __m128 mul2 = _mm_mul_ps(q_shuffled, _mm_set_ps(s, s, s, s));

  // Apply signs based on formula
  const __m128 signs = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);
  __m128 mul2_signed = _mm_mul_ps(mul2, signs);
  __m128 res_vec = _mm_add_ps(mul1, _mm_shuffle_ps(mul2_signed, mul2_signed, _MM_SHUFFLE(2, 3, 0, 1)));

  // Reorder to get [x, y, z, w]
  res_vec = _mm_shuffle_ps(res_vec, res_vec, _MM_SHUFFLE(2, 3, 1, 0));
  _mm_storeu_ps(result.v, res_vec);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c - q_z * s;
  result.v[1] = q_y * c + q_w * s;
  result.v[2] = q_z * c + q_x * s;
  result.v[3] = q_w * c - q_y * s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  v128_t q_vec = wasm_v128_load(q.v);

  // Create coefficients
  v128_t c_vec = wasm_f32x4_splat(c);
  v128_t s_vec = wasm_f32x4_splat(s);

  // Calculate c * q
  v128_t c_mul_q = wasm_f32x4_mul(q_vec, c_vec);

  // Create shuffled quaternion [q_z, q_w, q_x, q_y] for s multiplication
  v128_t q_shuffled = wasm_f32x4_shuffle(q_vec, q_vec, 2, 3, 0, 1);

  // Calculate s * [q_z, q_w, q_x, q_y] = [s*q_z, s*q_w, s*q_x, s*q_y]
  v128_t s_mul_shuffled = wasm_f32x4_mul(q_shuffled, s_vec);

  // Apply signs based on rotation formula: [1, 1, -1, -1]
  v128_t sign_mask = wasm_v128_const(0x00000000, 0x00000000, 0x80000000, 0x80000000);  // [0, 0, -0, -0]
  v128_t s_signed = wasm_v128_xor(s_mul_shuffled, sign_mask);

  // Add c*q and signed s*[q_z, q_w, q_x, q_y] to get [c*q_x-s*q_z, c*q_y+s*q_w, c*q_z-s*q_x, c*q_w-s*q_y]
  v128_t result_vec = wasm_f32x4_add(c_mul_q, s_signed);

  // Reorder to get [c*q_x-s*q_z, c*q_y+s*q_w, c*q_z+s*q_x, c*q_w-s*q_y] (correct for y rotation)
  result_vec = wasm_f32x4_shuffle(result_vec, result_vec, 0, 1, 2, 3);

  // Store result
  wasm_v128_store(result.v, result_vec);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // For RISC-V, we'll use a scalar approach for quaternion rotation
  // as the vector operations would be more complex to implement efficiently
  // for this specific operation
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c - q_z * s;
  result.v[1] = q_y * c + q_w * s;
  result.v[2] = q_z * c + q_x * s;
  result.v[3] = q_w * c - q_y * s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  __m128 q_vec = __lsx_vld(q.v, 0);

  // Create coefficients
  __m128 c_vec = __lsx_vldrepl_w(&c, 0);
  __m128 s_vec = __lsx_vldrepl_w(&s, 0);

  // Calculate c * q
  __m128 c_mul_q = __lsx_vfmul_s(q_vec, c_vec);

  // Create shuffled quaternion [q_z, q_w, q_x, q_y] for s multiplication
  __m128 q_shuffled = __lsx_vshuf4i_w(q_vec, 0x39); // [q_z, q_w, q_x, q_y]

  // Calculate s * [q_z, q_w, q_x, q_y] = [s*q_z, s*q_w, s*q_x, s*q_y]
  __m128 s_mul_shuffled = __lsx_vfmul_s(q_shuffled, s_vec);

  // Apply signs based on rotation formula: [1, 1, -1, -1]
  __m128 sign_mask = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&(float){0.0f}, 0), (1u << 31), 2), (1u << 31), 3);
  __m128 s_signed = __lsx_vxor_v(s_mul_shuffled, sign_mask);

  // Add c*q and signed s*[q_z, q_w, q_x, q_y] to get [c*q_x-s*q_z, c*q_y+s*q_w, c*q_z+s*q_x, c*q_w-s*q_y]
  __m128 result_vec = __lsx_vfadd_s(c_mul_q, s_signed);

  // Store result
  __lsx_vst(result_vec, result.v, 0);

#else
  // Scalar implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c - q_z * s;
  result.v[1] = q_y * c + q_w * s;
  result.v[2] = q_z * c + q_x * s;
  result.v[3] = q_w * c - q_y * s;
#endif

  return result;
}

// Quat rotate_z
WMATH_TYPE(Quat)
WMATH_ROTATE_Z(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  __m128 q_vec = _mm_loadu_ps(q.v);

  // Create rotation quaternion b = [0, 0, s, c]
  _mm_set_ps(c, s, 0.0f, 0.0f);

  // For rotation around z-axis:
  // result.x = q.x * b.w + q.y * b.z = q[0] * b[3] + q[1] * b[2]
  // result.y = q.y * b.w - q.x * b.z = q[1] * b[3] - q[0] * b[2]
  // result.z = q.z * b.w + q.w * b.z = q[2] * b[3] + q[3] * b[2]
  // result.w = q.w * b.w - q.z * b.z = q[3] * b[3] - q[2] * b[2]

  __m128 mul1 = _mm_mul_ps(q_vec, _mm_set_ps(c, c, c, c));
  __m128 q_shuffled = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(2, 3, 0, 1)); // [y, x, w, z]
  __m128 mul2 = _mm_mul_ps(q_shuffled, _mm_set_ps(s, s, s, s));

  // Apply signs based on formula
  __m128 signs = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
  __m128 mul2_signed = _mm_mul_ps(mul2, signs);
  __m128 res_vec = _mm_add_ps(mul1, _mm_shuffle_ps(mul2_signed, mul2_signed, _MM_SHUFFLE(3, 2, 1, 0)));

  // Reorder to get [x, y, z, w]
  res_vec = _mm_shuffle_ps(res_vec, res_vec, _MM_SHUFFLE(3, 2, 0, 1));
  _mm_storeu_ps(result.v, res_vec);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c + q_y * s;
  result.v[1] = q_y * c - q_x * s;
  result.v[2] = q_z * c + q_w * s;
  result.v[3] = q_w * c - q_z * s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  v128_t q_vec = wasm_v128_load(q.v);

  // Create coefficients
  v128_t c_vec = wasm_f32x4_splat(c);
  v128_t s_vec = wasm_f32x4_splat(s);

  // Calculate c * q
  v128_t c_mul_q = wasm_f32x4_mul(q_vec, c_vec);

  // Create shuffled quaternion [q_y, q_x, q_w, q_z] for s multiplication
  v128_t q_shuffled = wasm_f32x4_shuffle(q_vec, q_vec, 1, 0, 3, 2);

  // Calculate s * [q_y, q_x, q_w, q_z] = [s*q_y, s*q_x, s*q_w, s*q_z]
  v128_t s_mul_shuffled = wasm_f32x4_mul(q_shuffled, s_vec);

  // Apply signs based on rotation formula: [1, -1, 1, -1]
  v128_t sign_mask = wasm_v128_const(0x00000000, 0x80000000, 0x00000000, 0x80000000);  // [0, -0, 0, -0]
  v128_t s_signed = wasm_v128_xor(s_mul_shuffled, sign_mask);

  // Add c*q and signed s*[q_y, q_x, q_w, q_z] to get [c*q_x+s*q_y, c*q_y-s*q_x, c*q_z+s*q_w, c*q_w-s*q_z]
  v128_t result_vec = wasm_f32x4_add(c_mul_q, s_signed);

  // Store result
  wasm_v128_store(result.v, result_vec);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // For RISC-V, we'll use a scalar approach for quaternion rotation
  // as the vector operations would be more complex to implement efficiently
  // for this specific operation
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c + q_y * s;
  result.v[1] = q_y * c - q_x * s;
  result.v[2] = q_z * c + q_w * s;
  result.v[3] = q_w * c - q_z * s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);

  // Load quaternion
  __m128 q_vec = __lsx_vld(q.v, 0);

  // Create coefficients
  __m128 c_vec = __lsx_vldrepl_w(&c, 0);
  __m128 s_vec = __lsx_vldrepl_w(&s, 0);

  // Calculate c * q
  __m128 c_mul_q = __lsx_vfmul_s(q_vec, c_vec);

  // Create shuffled quaternion [q_y, q_x, q_w, q_z] for s multiplication
  __m128 q_shuffled = __lsx_vshuf4i_w(q_vec, 0x93); // [q_y, q_x, q_w, q_z]

  // Calculate s * [q_y, q_x, q_w, q_z] = [s*q_y, s*q_x, s*q_w, s*q_z]
  __m128 s_mul_shuffled = __lsx_vfmul_s(q_shuffled, s_vec);

  // Apply signs based on rotation formula: [1, -1, 1, -1]
  __m128 sign_mask = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&(float){0.0f}, 0), (1u << 31), 1), (1u << 31), 3);
  __m128 s_signed = __lsx_vxor_v(s_mul_shuffled, sign_mask);

  // Add c*q and signed s*[q_y, q_x, q_w, q_z] to get [c*q_x+s*q_y, c*q_y-s*q_x, c*q_z+s*q_w, c*q_w-s*q_z]
  __m128 result_vec = __lsx_vfadd_s(c_mul_q, s_signed);

  // Store result
  __lsx_vst(result_vec, result.v, 0);

#else
  // Scalar implementation
  float half_angle = angleInRadians * 0.5f;
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  float b_z = sinf(half_angle);
  float b_w = cosf(half_angle);
  result.v[0] = q_x * b_w + q_y * b_z;
  result.v[1] = q_y * b_w - q_x * b_z;
  result.v[2] = q_z * b_w + q_w * b_z;
  result.v[3] = q_w * b_w - q_z * b_z;
#endif

  return result;
}

// Mat3 rotate
WMATH_TYPE(Mat3)
WMATH_ROTATE(Mat3)
(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first two rows
  __m128 row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  __m128 row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]

  // Create coefficients
  __m128 c_vec = _mm_set1_ps(c);
  __m128 s_vec = _mm_set1_ps(s);

  // Calculate new first row: c * row0 + s * row1
  __m128 new_row0 = _mm_add_ps(
    _mm_mul_ps(c_vec, row0),
    _mm_mul_ps(s_vec, row1)
  );

  // Calculate new second row: c * row1 - s * row0
  __m128 new_row1 = _mm_sub_ps(
    _mm_mul_ps(c_vec, row1),
    _mm_mul_ps(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 1, new_row1);

  // Copy third row if needed
  if (!_mm_movemask_ps(_mm_cmpeq_ps(new_row0, row0)) ||
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row1, row1))) {
    wcn_mat3_set_row(&newDst, 2, wcn_mat3_get_row(&m, 2));
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first two rows
  float32x4_t row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  float32x4_t row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]

  // Create coefficients
  float32x4_t c_vec = vdupq_n_f32(c);
  float32x4_t s_vec = vdupq_n_f32(s);

  // Calculate new first row: c * row0 + s * row1
  float32x4_t new_row0 = vaddq_f32(
    vmulq_f32(c_vec, row0),
    vmulq_f32(s_vec, row1)
  );

  // Calculate new second row: c * row1 - s * row0
  float32x4_t new_row1 = vsubq_f32(
    vmulq_f32(c_vec, row1),
    vmulq_f32(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 1, new_row1);

  // Simple check for equality (simplified version of the original logic)
  uint32x4_t eq0 = vceqq_f32(new_row0, row0);
  uint32x4_t eq1 = vceqq_f32(new_row1, row1);
  if (!(vgetq_lane_u32(eq0, 0) && vgetq_lane_u32(eq0, 1) && vgetq_lane_u32(eq0, 2) &&
        vgetq_lane_u32(eq1, 0) && vgetq_lane_u32(eq1, 1) && vgetq_lane_u32(eq1, 2))) {
    wcn_mat3_set_row(&newDst, 2, wcn_mat3_get_row(&m, 2));
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first two rows
  v128_t row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  v128_t row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]

  // Create coefficients
  v128_t c_vec = wasm_f32x4_splat(c);
  v128_t s_vec = wasm_f32x4_splat(s);

  // Calculate new first row: c * row0 + s * row1
  v128_t new_row0 = wasm_f32x4_add(
    wasm_f32x4_mul(c_vec, row0),
    wasm_f32x4_mul(s_vec, row1)
  );

  // Calculate new second row: c * row1 - s * row0
  v128_t new_row1 = wasm_f32x4_sub(
    wasm_f32x4_mul(c_vec, row1),
    wasm_f32x4_mul(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 1, new_row1);

  // WASM doesn't have the same comparison logic, so we'll check by default
  wcn_mat3_set_row(&newDst, 2, wcn_mat3_get_row(&m, 2));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // For RISC-V, we'll use a scalar approach for matrix rotation
  // as complex SIMD operations for 3x3 matrices can be less efficient for this architecture
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m10 = m.m[1 * 4 + 0];
  float m11 = m.m[1 * 4 + 1];
  float m12 = m.m[1 * 4 + 2];

  newDst.m[0] = c * m00 + s * m10;
  newDst.m[1] = c * m01 + s * m11;
  newDst.m[2] = c * m02 + s * m12;

  newDst.m[4] = c * m10 - s * m00;
  newDst.m[5] = c * m11 - s * m01;
  newDst.m[6] = c * m12 - s * m02;

  newDst.m[8] = m.m[8];
  newDst.m[9] = m.m[9];
  newDst.m[10] = m.m[10];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first two rows
  __m128 row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  __m128 row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]

  // Create coefficients
  __m128 c_vec = __lsx_vldrepl_w(&c, 0);
  __m128 s_vec = __lsx_vldrepl_w(&s, 0);

  // Calculate new first row: c * row0 + s * row1
  __m128 new_row0 = __lsx_vfadd_s(
    __lsx_vfmul_s(c_vec, row0),
    __lsx_vfmul_s(s_vec, row1)
  );

  // Calculate new second row: c * row1 - s * row0
  __m128 new_row1 = __lsx_vfsub_s(
    __lsx_vfmul_s(c_vec, row1),
    __lsx_vfmul_s(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 1, new_row1);

  // Simple check by copying the third row
  wcn_mat3_set_row(&newDst, 2, wcn_mat3_get_row(&m, 2));

#else
  // Scalar implementation (original code)
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m10 = m.m[1 * 4 + 0];
  float m11 = m.m[1 * 4 + 1];
  float m12 = m.m[1 * 4 + 2];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 + s * m10;
  newDst.m[1] = c * m01 + s * m11;
  newDst.m[2] = c * m02 + s * m12;

  newDst.m[4] = c * m10 - s * m00;
  newDst.m[5] = c * m11 - s * m01;
  newDst.m[6] = c * m12 - s * m02;

  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }
#endif

  return newDst;
}

// Mat3 rotate x
WMATH_TYPE(Mat3)
WMATH_ROTATE_X(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load second and third rows
  __m128 row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  __m128 row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  __m128 c_vec = _mm_set1_ps(c);
  __m128 s_vec = _mm_set1_ps(s);

  // Calculate new second row: c * row1 + s * row2
  __m128 new_row1 = _mm_add_ps(
    _mm_mul_ps(c_vec, row1),
    _mm_mul_ps(s_vec, row2)
  );

  // Calculate new third row: c * row2 - s * row1
  __m128 new_row2 = _mm_sub_ps(
    _mm_mul_ps(c_vec, row2),
    _mm_mul_ps(s_vec, row1)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 1, new_row1);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // Copy first row if needed
  if (!_mm_movemask_ps(_mm_cmpeq_ps(new_row1, row1)) ||
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row2, row2))) {
    wcn_mat3_set_row(&newDst, 0, wcn_mat3_get_row(&m, 0));
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load second and third rows
  float32x4_t row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  float32x4_t row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  float32x4_t c_vec = vdupq_n_f32(c);
  float32x4_t s_vec = vdupq_n_f32(s);

  // Calculate new second row: c * row1 + s * row2
  float32x4_t new_row1 = vaddq_f32(
    vmulq_f32(c_vec, row1),
    vmulq_f32(s_vec, row2)
  );

  // Calculate new third row: c * row2 - s * row1
  float32x4_t new_row2 = vsubq_f32(
    vmulq_f32(c_vec, row2),
    vmulq_f32(s_vec, row1)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 1, new_row1);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // Simple check for equality (simplified version of the original logic)
  uint32x4_t eq1 = vceqq_f32(new_row1, row1);
  uint32x4_t eq2 = vceqq_f32(new_row2, row2);
  if (!(vgetq_lane_u32(eq1, 0) && vgetq_lane_u32(eq1, 1) && vgetq_lane_u32(eq1, 2) &&
        vgetq_lane_u32(eq2, 0) && vgetq_lane_u32(eq2, 1) && vgetq_lane_u32(eq2, 2))) {
    wcn_mat3_set_row(&newDst, 0, wcn_mat3_get_row(&m, 0));
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load second and third rows
  v128_t row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  v128_t row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  v128_t c_vec = wasm_f32x4_splat(c);
  v128_t s_vec = wasm_f32x4_splat(s);

  // Calculate new second row: c * row1 + s * row2
  v128_t new_row1 = wasm_f32x4_add(
    wasm_f32x4_mul(c_vec, row1),
    wasm_f32x4_mul(s_vec, row2)
  );

  // Calculate new third row: c * row2 - s * row1
  v128_t new_row2 = wasm_f32x4_sub(
    wasm_f32x4_mul(c_vec, row2),
    wasm_f32x4_mul(s_vec, row1)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 1, new_row1);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // WASM doesn't have the same comparison logic, so we'll check by default
  wcn_mat3_set_row(&newDst, 0, wcn_mat3_get_row(&m, 0));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // For RISC-V, we'll use a scalar approach for matrix rotation
  float m_10 = m.m[4];
  float m_11 = m.m[5];
  float m_12 = m.m[6];
  float m_20 = m.m[8];
  float m_21 = m.m[9];
  float m_22 = m.m[10];

  newDst.m[4] = c * m_10 + s * m_20;
  newDst.m[5] = c * m_11 + s * m_21;
  newDst.m[6] = c * m_12 + s * m_22;
  newDst.m[8] = c * m_20 - s * m_10;
  newDst.m[9] = c * m_21 - s * m_11;
  newDst.m[10] = c * m_22 - s * m_12;

  newDst.m[0] = m.m[0];
  newDst.m[1] = m.m[1];
  newDst.m[2] = m.m[2];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load second and third rows
  __m128 row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  __m128 row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  __m128 c_vec = __lsx_vldrepl_w(&c, 0);
  __m128 s_vec = __lsx_vldrepl_w(&s, 0);

  // Calculate new second row: c * row1 + s * row2
  __m128 new_row1 = __lsx_vfadd_s(
    __lsx_vfmul_s(c_vec, row1),
    __lsx_vfmul_s(s_vec, row2)
  );

  // Calculate new third row: c * row2 - s * row1
  __m128 new_row2 = __lsx_vfsub_s(
    __lsx_vfmul_s(c_vec, row2),
    __lsx_vfmul_s(s_vec, row1)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 1, new_row1);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // Simple check by copying the first row
  wcn_mat3_set_row(&newDst, 0, wcn_mat3_get_row(&m, 0));

#else
  // Scalar implementation (original code)
  float m_10 = m.m[4];
  float m_11 = m.m[5];
  float m_12 = m.m[6];
  float m_20 = m.m[8];
  float m_21 = m.m[9];
  float m_22 = m.m[10];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[4] = c * m_10 + s * m_20;
  newDst.m[5] = c * m_11 + s * m_21;
  newDst.m[6] = c * m_12 + s * m_22;
  newDst.m[8] = c * m_20 - s * m_10;
  newDst.m[9] = c * m_21 - s * m_11;
  newDst.m[10] = c * m_22 - s * m_12;
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
  }
#endif

  return newDst;
}

// Mat3 rotate y
WMATH_TYPE(Mat3)
WMATH_ROTATE_Y(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first and third rows
  __m128 row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  __m128 row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  __m128 c_vec = _mm_set1_ps(c);
  __m128 s_vec = _mm_set1_ps(s);

  // Calculate new first row: c * row0 - s * row2
  __m128 new_row0 = _mm_sub_ps(
    _mm_mul_ps(c_vec, row0),
    _mm_mul_ps(s_vec, row2)
  );

  // Calculate new third row: c * row2 + s * row0
  __m128 new_row2 = _mm_add_ps(
    _mm_mul_ps(c_vec, row2),
    _mm_mul_ps(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // Copy second row if needed
  if (!_mm_movemask_ps(_mm_cmpeq_ps(new_row0, row0)) ||
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row2, row2))) {
    wcn_mat3_set_row(&newDst, 1, wcn_mat3_get_row(&m, 1));
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first and third rows
  float32x4_t row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  float32x4_t row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  float32x4_t c_vec = vdupq_n_f32(c);
  float32x4_t s_vec = vdupq_n_f32(s);

  // Calculate new first row: c * row0 - s * row2
  float32x4_t new_row0 = vsubq_f32(
    vmulq_f32(c_vec, row0),
    vmulq_f32(s_vec, row2)
  );

  // Calculate new third row: c * row2 + s * row0
  float32x4_t new_row2 = vaddq_f32(
    vmulq_f32(c_vec, row2),
    vmulq_f32(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // Simple check for equality (simplified version of the original logic)
  uint32x4_t eq0 = vceqq_f32(new_row0, row0);
  uint32x4_t eq2 = vceqq_f32(new_row2, row2);
  if (!(vgetq_lane_u32(eq0, 0) && vgetq_lane_u32(eq0, 1) && vgetq_lane_u32(eq0, 2) &&
        vgetq_lane_u32(eq2, 0) && vgetq_lane_u32(eq2, 1) && vgetq_lane_u32(eq2, 2))) {
    wcn_mat3_set_row(&newDst, 1, wcn_mat3_get_row(&m, 1));
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first and third rows
  v128_t row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  v128_t row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  v128_t c_vec = wasm_f32x4_splat(c);
  v128_t s_vec = wasm_f32x4_splat(s);

  // Calculate new first row: c * row0 - s * row2
  v128_t new_row0 = wasm_f32x4_sub(
    wasm_f32x4_mul(c_vec, row0),
    wasm_f32x4_mul(s_vec, row2)
  );

  // Calculate new third row: c * row2 + s * row0
  v128_t new_row2 = wasm_f32x4_add(
    wasm_f32x4_mul(c_vec, row2),
    wasm_f32x4_mul(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // WASM doesn't have the same comparison logic, so we'll check by default
  wcn_mat3_set_row(&newDst, 1, wcn_mat3_get_row(&m, 1));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // For RISC-V, we'll use a scalar approach for matrix rotation
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m20 = m.m[2 * 4 + 0];
  float m21 = m.m[2 * 4 + 1];
  float m22 = m.m[2 * 4 + 2];

  newDst.m[0] = c * m00 - s * m20;
  newDst.m[1] = c * m01 - s * m21;
  newDst.m[2] = c * m02 - s * m22;
  newDst.m[8] = c * m20 + s * m00;
  newDst.m[9] = c * m21 + s * m01;
  newDst.m[10] = c * m22 + s * m02;

  newDst.m[4] = m.m[4];
  newDst.m[5] = m.m[5];
  newDst.m[6] = m.m[6];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  // Load first and third rows
  __m128 row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  __m128 row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]

  // Create coefficients
  __m128 c_vec = __lsx_vldrepl_w(&c, 0);
  __m128 s_vec = __lsx_vldrepl_w(&s, 0);

  // Calculate new first row: c * row0 - s * row2
  __m128 new_row0 = __lsx_vfsub_s(
    __lsx_vfmul_s(c_vec, row0),
    __lsx_vfmul_s(s_vec, row2)
  );

  // Calculate new third row: c * row2 + s * row0
  __m128 new_row2 = __lsx_vfadd_s(
    __lsx_vfmul_s(c_vec, row2),
    __lsx_vfmul_s(s_vec, row0)
  );

  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 2, new_row2);

  // Simple check by copying the second row
  wcn_mat3_set_row(&newDst, 1, wcn_mat3_get_row(&m, 1));

#else
  // Scalar implementation (original code)
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m20 = m.m[2 * 4 + 0];
  float m21 = m.m[2 * 4 + 1];
  float m22 = m.m[2 * 4 + 2];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 - s * m20;
  newDst.m[1] = c * m01 - s * m21;
  newDst.m[2] = c * m02 - s * m22;
  newDst.m[8] = c * m20 + s * m00;
  newDst.m[9] = c * m21 + s * m01;
  newDst.m[10] = c * m22 + s * m02;

  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
  }
#endif

  return newDst;
}

// Mat3 rotate z
WMATH_TYPE(Mat3)
WMATH_ROTATE_Z(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  return WMATH_ROTATE(Mat3)(m, angleInRadians);
}

// Mat3 rotation
WMATH_TYPE(Mat3)
WMATH_ROTATION(Mat3)(float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[0] = c;
  newDst.m[1] = s;
  // newDst.m[2] = 0;
  newDst.m[4] = -s;
  newDst.m[5] = c;
  // newDst.m[6] = 0;
  // newDst.m[8] = 0;
  // newDst.m[9] = 0;
  newDst.m[10] = 1;
  return newDst;
}

// Mat3 rotation x
WMATH_TYPE(Mat3)
WMATH_ROTATION_X(Mat3)(float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[0] = 1;
  // newDst.m[1] = 0;
  // newDst.m[2] = 0;
  // newDst.m[4] = 0;
  newDst.m[5] = c;
  newDst.m[6] = s;
  // newDst.m[8] = 0;
  newDst.m[9] = -s;
  newDst.m[10] = c;
  return newDst;
}

// Mat3 rotation y
WMATH_TYPE(Mat3)
WMATH_ROTATION_Y(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[0] = c;
  // newDst.m[1] = 0;
  newDst.m[2] = -s;
  // newDst.m[4] = 0;
  newDst.m[5] = 1;
  // newDst.m[6] = 0;
  newDst.m[8] = s;
  // newDst.m[9] = 0;
  newDst.m[10] = c;

  return newDst;
}

// Mat3 rotation z
WMATH_TYPE(Mat3)
WMATH_ROTATION_Z(Mat3)(float angleInRadians) {
  return WMATH_ROTATION(Mat3)(angleInRadians);
}

// Mat3 get_axis
/**
 * Returns an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y,
 * @returns The axis component of m.
 */
WMATH_TYPE(Vec2)
WMATH_CALL(Mat3, get_axis)
(WMATH_TYPE(Mat3) m, int axis) {
  WMATH_TYPE(Vec2) result;
  int off = axis * 4;
  result.v[0] = m.m[off + 0];
  result.v[1] = m.m[off + 1];
  return result;
}
// Mat3 set_axis
/**
 * Sets an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param v - the axis vector
 * @param axis - Axis  0 = x, 1 = y;
 * @returns The matrix with axis set.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, set_axis)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v, int axis) {
  WMATH_TYPE(Mat3) newDst = WMATH_COPY(Mat3)(m);
  int off = axis * 4;
  newDst.m[off + 0] = v.v[0];
  newDst.m[off + 1] = v.v[1];
  return newDst;
}

// Mat3 get_scaling
WMATH_TYPE(Vec2)
WMATH_CALL(Mat3, get_scaling)
(const WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec2) result;
  const float xx = m.m[0];
  const float xy = m.m[1];
  const float yx = m.m[4];
  const float yy = m.m[5];
  result.v[0] = sqrtf(xx * xx + xy * xy);
  result.v[1] = sqrtf(yx * yx + yy * yy);
  return result;
}

// Mat3 get_3D_scaling
WMATH_TYPE(Vec3)
WMATH_CALL(Mat3, get_3D_scaling)
(const WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec3) result;
  const float xx = m.m[0];
  const float xy = m.m[1];
  const float xz = m.m[2];
  const float yx = m.m[4];
  const float yy = m.m[5];
  const float yz = m.m[6];
  const float zx = m.m[8];
  const float zy = m.m[9];
  const float zz = m.m[10];

  result.v[0] = sqrtf(xx * xx + xy * xy + xz * xz);
  result.v[1] = sqrtf(yx * yx + yy * yy + yz * yz);
  result.v[2] = sqrtf(zx * zx + zy * zy + zz * zz);

  return result;
}

// Mat3 get_translation
WMATH_TYPE(Vec2)
WMATH_GET_TRANSLATION(Mat3)
(WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec2) result;
  result.v[0] = m.m[8];
  result.v[1] = m.m[9];
  return result;
}

// Mat3 set_translation
/**
 * Sets the translation component of a 3-by-3 matrix to the given
 * vector.
 * @param m - The matrix.
 * @param v - The vector.
 * @returns The matrix with translation set.
 */
WMATH_TYPE(Mat3)
WMATH_SET_TRANSLATION(Mat3)
(const WMATH_TYPE(Mat3) m, const WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_IDENTITY(Mat3)();
  if (!WMATH_EQUALS(Mat3)(m, newDst)) {
    newDst.m[0] = v.v[0];
    newDst.m[1] = v.v[1];
    newDst.m[2] = v.v[2];
    newDst.m[4] = v.v[4];
    newDst.m[5] = v.v[5];
    newDst.m[6] = v.v[6];
  }
  newDst.m[8] = v.v[0];
  newDst.m[9] = v.v[1];
  newDst.m[10] = 1;

  return newDst;
}

// Mat3 translation
WMATH_TYPE(Mat3)
WMATH_TRANSLATION(Mat3)
(WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - create 2D translation matrix efficiently
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);

  // Create identity matrix with translation in third row
  // Row0: [1, 0, 0, pad]
  __m128 row0 = _mm_move_ss(vec_one, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, 1, 0, pad]
  __m128 row1 = _mm_move_ss(vec_zero, vec_one);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Row2: [v.x, v.y, 1, pad]
  __m128 row2 = _mm_move_ss(vec_v, vec_one);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - create 2D translation matrix efficiently
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Create identity matrix with translation in third row
  float32x4_t row0 = vec_zero;
  row0 = vsetq_lane_f32(1.0f, row0, 0);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(1.0f, row1, 1);

  float32x4_t row2 = vec_v;
  row2 = vsetq_lane_f32(1.0f, row2, 2);

  // Store results
  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - create 2D translation matrix efficiently
  v128_t vec_v = wasm_v128_load(v.v);  // Load v.x, v.y and potentially other values
  v128_t vec_zero = wasm_f32x4_splat(0.0f);
  v128_t vec_one = wasm_f32x4_splat(1.0f);

  // Create identity matrix with translation in third row
  // Row0: [1, 0, 0, 0]
  v128_t row0 = wasm_v128_load(&newDst.m[0]); // Load default to ensure we don't change other values
  row0 = wasm_f32x4_replace_lane(row0, 0, 1.0f);    // Set index 0 to 1.0
  row0 = wasm_f32x4_replace_lane(row0, 1, 0.0f);    // Set index 1 to 0.0
  row0 = wasm_f32x4_replace_lane(row0, 2, 0.0f);    // Set index 2 to 0.0
  row0 = wasm_f32x4_replace_lane(row0, 3, 0.0f);    // Set index 3 to 0.0

  // Row1: [0, 1, 0, 0]
  v128_t row1 = wasm_v128_load(&newDst.m[4]); // Load default to ensure we don't change other values
  row1 = wasm_f32x4_replace_lane(row1, 0, 0.0f);    // Set index 0 to 0.0
  row1 = wasm_f32x4_replace_lane(row1, 1, 1.0f);    // Set index 1 to 1.0
  row1 = wasm_f32x4_replace_lane(row1, 2, 0.0f);    // Set index 2 to 0.0
  row1 = wasm_f32x4_replace_lane(row1, 3, 0.0f);    // Set index 3 to 0.0

  // Row2: [v.x, v.y, 1, 0]
  v128_t row2 = wasm_f32x4_replace_lane(vec_zero, 0, v.v[0]);  // Set index 0 to v.x
  row2 = wasm_f32x4_replace_lane(row2, 1, v.v[1]);             // Set index 1 to v.y
  row2 = wasm_f32x4_replace_lane(row2, 2, 1.0f);               // Set index 2 to 1.0

  // Store results
  wasm_v128_store(&newDst.m[0], row0);
  wasm_v128_store(&newDst.m[4], row1);
  wasm_v128_store(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation - create 2D translation matrix efficiently
  size_t vl = __riscv_vsetvli(4, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);

  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, vl);
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vec_one = __riscv_vfmv_v_f_f32m1(1.0f, vl);

  // Create identity matrix with translation in third row
  // Row0: [1, 0, 0, 0]
  vfloat32m1_t row0 = __riscv_vslideup_vx_f32m1(__riscv_vslideup_vx_f32m1(vec_zero, 1.0f, 0, vl), 0.0f, 1, vl);

  // Row1: [0, 1, 0, 0]
  vfloat32m1_t row1 = __riscv_vslideup_vx_f32m1(__riscv_vslideup_vx_f32m1(vec_zero, 0.0f, 0, vl), 1.0f, 1, vl);

  // Row2: [v.x, v.y, 1, 0]
  vfloat32m1_t row2 = __riscv_vslideup_vx_f32m1(__riscv_vslideup_vx_f32m1(vec_v, 1.0f, 2, vl), 0.0f, 3, vl);

  // Store results
  __riscv_vse32_v_f32m1(&newDst.m[0], row0, vl);
  __riscv_vse32_v_f32m1(&newDst.m[4], row1, vl);
  __riscv_vse32_v_f32m1(&newDst.m[8], row2, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - create 2D translation matrix efficiently
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 vec_zero = __lsx_vffint_s_w(__lsx_vldreplgr2vr_w(0));
  __m128 vec_one = __lsx_vldrepl_w(&WMATH_ONE, 0);

  // Create identity matrix with translation in third row
  // Row0: [1, 0, 0, 0]
  __m128 row0 = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&WMATH_ZERO, 0), 1.0f, 0), 0.0f, 1), 0.0f, 2);
  row0 = __lsx_vinsgr2vr_w(row0, 0.0f, 3);

  // Row1: [0, 1, 0, 0]
  __m128 row1 = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&WMATH_ZERO, 0), 0.0f, 0), 1.0f, 1), 0.0f, 2);
  row1 = __lsx_vinsgr2vr_w(row1, 0.0f, 3);

  // Row2: [v.x, v.y, 1, 0]
  __m128 row2 = __lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vinsgr2vr_w(__lsx_vldrepl_w(&WMATH_ZERO, 0), v.v[0], 0), v.v[1], 1), 1.0f, 2);
  row2 = __lsx_vinsgr2vr_w(row2, 0.0f, 3);

  // Store results
  __lsx_vst(row0, &newDst.m[0], 0);
  __lsx_vst(row1, &newDst.m[4], 0);
  __lsx_vst(row2, &newDst.m[8], 0);

#else
  // Scalar fallback - direct assignment is more efficient than memset
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat3)));
  newDst.m[0] = 1.0f;
  newDst.m[5] = 1.0f;
  newDst.m[8] = v.v[0];
  newDst.m[9] = v.v[1];
  newDst.m[10] = 1.0f;
#endif

  return newDst;
}

// translate
/**
 * Translates the given 3-by-3 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by which to translate.
 * @returns The translated matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, translate)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized 2D matrix translation
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 row0 = wcn_mat3_get_row(&m, 0);
  __m128 row1 = wcn_mat3_get_row(&m, 1);
  __m128 row2 = wcn_mat3_get_row(&m, 2);

  // Copy the first two rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    wcn_mat3_set_row(&newDst, 0, row0);
    wcn_mat3_set_row(&newDst, 1, row1);
  }

  // Calculate translation components using SIMD
  // Extract x, y components from translation vector
  __m128 v_x = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 v_y = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(1, 1, 1, 1));

  // Calculate dot products for each translation component
  __m128 dot_x = _mm_mul_ps(row0, v_x);
  __m128 dot_y = _mm_mul_ps(row1, v_y);

  // Sum the dot products and add original translation
  __m128 sum_xy = _mm_add_ps(dot_x, dot_y);
  __m128 trans_sum = _mm_add_ps(sum_xy, row2);

  // Store the translation row
  wcn_mat3_set_row(&newDst, 2, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized 2D matrix translation
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t row0 = wcn_mat3_get_row(&m, 0);
  float32x4_t row1 = wcn_mat3_get_row(&m, 1);
  float32x4_t row2 = wcn_mat3_get_row(&m, 2);

  // Copy the first two rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    wcn_mat3_set_row(&newDst, 0, row0);
    wcn_mat3_set_row(&newDst, 1, row1);
  }

  // Calculate translation components using SIMD
  // Extract x, y components from translation vector
  float32x4_t v_x = vdupq_lane_f32(vget_low_f32(vec_v), 0);
  float32x4_t v_y = vdupq_lane_f32(vget_low_f32(vec_v), 1);

  // Calculate dot products for each translation component
  float32x4_t dot_x = vmulq_f32(row0, v_x);
  float32x4_t dot_y = vmulq_f32(row1, v_y);

  // Sum the dot products and add original translation
  float32x4_t sum_xy = vaddq_f32(dot_x, dot_y);
  float32x4_t trans_sum = vaddq_f32(sum_xy, row2);

  // Store the translation row
  wcn_mat3_set_row(&newDst, 2, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - optimized 2D matrix translation
  v128_t vec_v = wasm_v128_load(v.v);
  v128_t row0 = wcn_mat3_get_row(&m, 0);
  v128_t row1 = wcn_mat3_get_row(&m, 1);
  v128_t row2 = wcn_mat3_get_row(&m, 2);

  // Copy the first two rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    wcn_mat3_set_row(&newDst, 0, row0);
    wcn_mat3_set_row(&newDst, 1, row1);
  }

  // Calculate translation components using SIMD
  // Extract x, y components from translation vector
  v128_t v_x = wasm_f32x4_splat(wasm_f32x4_extract_lane(vec_v, 0));
  v128_t v_y = wasm_f32x4_splat(wasm_f32x4_extract_lane(vec_v, 1));

  // Calculate dot products for each translation component
  v128_t dot_x = wasm_f32x4_mul(row0, v_x);
  v128_t dot_y = wasm_f32x4_mul(row1, v_y);

  // Sum the dot products and add original translation
  v128_t sum_xy = wasm_f32x4_add(dot_x, dot_y);
  v128_t trans_sum = wasm_f32x4_add(sum_xy, row2);

  // Store the translation row
  wcn_mat3_set_row(&newDst, 2, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation - optimized 2D matrix translation
  size_t vl = __riscv_vsetvli(4, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, vl);
  vfloat32m1_t row0 = wcn_mat3_get_row(&m, 0);
  vfloat32m1_t row1 = wcn_mat3_get_row(&m, 1);
  vfloat32m1_t row2 = wcn_mat3_get_row(&m, 2);

  // Copy the first two rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    wcn_mat3_set_row(&newDst, 0, row0);
    wcn_mat3_set_row(&newDst, 1, row1);
  }

  // Calculate translation components using SIMD
  // Extract x, y components from translation vector
  vfloat32m1_t v_x = __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(vec_v, 0), vl), vl);
  vfloat32m1_t v_y = __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(vec_v, 1), vl), vl);

  // Calculate dot products for each translation component
  vfloat32m1_t dot_x = __riscv_vfmul_vv_f32m1(row0, v_x, vl);
  vfloat32m1_t dot_y = __riscv_vfmul_vv_f32m1(row1, v_y, vl);

  // Sum the dot products and add original translation
  vfloat32m1_t sum_xy = __riscv_vfadd_vv_f32m1(dot_x, dot_y, vl);
  vfloat32m1_t trans_sum = __riscv_vfadd_vv_f32m1(sum_xy, row2, vl);

  // Store the translation row
  wcn_mat3_set_row(&newDst, 2, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - optimized 2D matrix translation
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 row0 = wcn_mat3_get_row(&m, 0);
  __m128 row1 = wcn_mat3_get_row(&m, 1);
  __m128 row2 = wcn_mat3_get_row(&m, 2);

  // Copy the first two rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    wcn_mat3_set_row(&newDst, 0, row0);
    wcn_mat3_set_row(&newDst, 1, row1);
  }

  // Calculate translation components using SIMD
  // Extract x, y components from translation vector
  __m128 v_x = __lsx_vreplvei_w(vec_v, 0);  // Extract x component and broadcast
  __m128 v_y = __lsx_vreplvei_w(vec_v, 1);  // Extract y component and broadcast

  // Calculate dot products for each translation component
  __m128 dot_x = __lsx_vfmul_s(row0, v_x);
  __m128 dot_y = __lsx_vfmul_s(row1, v_y);

  // Sum the dot products and add original translation
  __m128 sum_xy = __lsx_vfadd_s(dot_x, dot_y);
  __m128 trans_sum = __lsx_vfadd_s(sum_xy, row2);

  // Store the translation row
  wcn_mat3_set_row(&newDst, 2, trans_sum);

#else
  // Scalar fallback with optimized variable usage
  float v0 = v.v[0];
  float v1 = v.v[1];

  // Copy rotation/scaling part if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
  }

  // Calculate translation components with optimized ordering
  newDst.m[8] = m.m[0] * v0 + m.m[4] * v1 + m.m[8];
  newDst.m[9] = m.m[1] * v0 + m.m[5] * v1 + m.m[9];
  newDst.m[10] = m.m[2] * v0 + m.m[6] * v1 + m.m[10];
#endif

  return newDst;
}

// Mat4 axis_rotate
/**
 * Rotates the given 4-by-4 matrix around the given axis by the
 * given angle.
 * @param m - The matrix.
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, axis_rotate)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) axis, float angleInRadians) {
  // 我们不需要初始化为零，因为我们会覆盖所有行
  WMATH_TYPE(Mat4) newDst;

  // 1. 准备旋转矩阵系数 (Scalar部分)
  // 计算所有系数通常比向量化这部分更快，因为逻辑复杂且数据依赖性强
  float x = axis.v[0];
  float y = axis.v[1];
  float z = axis.v[2];

  // Normalize axis
  float lenSq = x*x + y*y + z*z;
  if (lenSq > 1e-6f) {
      float invLen = 1.0f / sqrtf(lenSq);
      x *= invLen;
      y *= invLen;
      z *= invLen;
  } else {
      // 轴无效，返回原矩阵或单位阵
      x = 0.0f; y = 0.0f; z = 1.0f;
  }

  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float oneMinusCosine = 1.0f - c;

  // Rotation matrix elements (R)
  // R 的第4行隐含为 0,0,0,1
  float r00 = xx + (1 - xx) * c;
  float r01 = x * y * oneMinusCosine + z * s;
  float r02 = x * z * oneMinusCosine - y * s;

  float r10 = x * y * oneMinusCosine - z * s;
  float r11 = yy + (1 - yy) * c;
  float r12 = y * z * oneMinusCosine + x * s;

  float r20 = x * z * oneMinusCosine + y * s;
  float r21 = y * z * oneMinusCosine - x * s;
  float r22 = zz + (1 - zz) * c;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE / AVX Implementation

  // Load original rows
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3); // Load row 3 directy

  // Row 0 calculation
  // newRow0 = r00*row0 + r01*row1 + r02*row2
  __m128 nr0 = _mm_mul_ps(_mm_set1_ps(r00), row0);
  nr0 = _mm_add_ps(nr0, _mm_mul_ps(_mm_set1_ps(r01), row1));
  nr0 = _mm_add_ps(nr0, _mm_mul_ps(_mm_set1_ps(r02), row2));

  // Row 1 calculation
  __m128 nr1 = _mm_mul_ps(_mm_set1_ps(r10), row0);
  nr1 = _mm_add_ps(nr1, _mm_mul_ps(_mm_set1_ps(r11), row1));
  nr1 = _mm_add_ps(nr1, _mm_mul_ps(_mm_set1_ps(r12), row2));

  // Row 2 calculation
  __m128 nr2 = _mm_mul_ps(_mm_set1_ps(r20), row0);
  nr2 = _mm_add_ps(nr2, _mm_mul_ps(_mm_set1_ps(r21), row1));
  nr2 = _mm_add_ps(nr2, _mm_mul_ps(_mm_set1_ps(r22), row2));

  // Store results
  wcn_mat4_set_row(&newDst, 0, nr0);
  wcn_mat4_set_row(&newDst, 1, nr1);
  wcn_mat4_set_row(&newDst, 2, nr2);
  wcn_mat4_set_row(&newDst, 3, row3); // Simply copy the translation/w row

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Implementation

  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // 使用 vmulq_n_f32 (Vector * Scalar) 和 vmlaq_n_f32 (Vector + Vector * Scalar)
  // 这是最紧凑的 NEON 写法

  // Row 0
  float32x4_t nr0 = vmulq_n_f32(row0, r00);
  nr0 = vmlaq_n_f32(nr0, row1, r01);
  nr0 = vmlaq_n_f32(nr0, row2, r02);

  // Row 1
  float32x4_t nr1 = vmulq_n_f32(row0, r10);
  nr1 = vmlaq_n_f32(nr1, row1, r11);
  nr1 = vmlaq_n_f32(nr1, row2, r12);

  // Row 2
  float32x4_t nr2 = vmulq_n_f32(row0, r20);
  nr2 = vmlaq_n_f32(nr2, row1, r21);
  nr2 = vmlaq_n_f32(nr2, row2, r22);

  wcn_mat4_set_row(&newDst, 0, nr0);
  wcn_mat4_set_row(&newDst, 1, nr1);
  wcn_mat4_set_row(&newDst, 2, nr2);
  wcn_mat4_set_row(&newDst, 3, row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD Implementation

  v128_t row0 = wcn_mat4_get_row_wasm(&m, 0);
  v128_t row1 = wcn_mat4_get_row_wasm(&m, 1);
  v128_t row2 = wcn_mat4_get_row_wasm(&m, 2);
  v128_t row3 = wcn_mat4_get_row_wasm(&m, 3);

  // Row 0
  v128_t nr0 = wasm_f32x4_mul(row0, wasm_f32x4_splat(r00));
  nr0 = wasm_f32x4_add(nr0, wasm_f32x4_mul(row1, wasm_f32x4_splat(r01)));
  nr0 = wasm_f32x4_add(nr0, wasm_f32x4_mul(row2, wasm_f32x4_splat(r02)));

  // Row 1
  v128_t nr1 = wasm_f32x4_mul(row0, wasm_f32x4_splat(r10));
  nr1 = wasm_f32x4_add(nr1, wasm_f32x4_mul(row1, wasm_f32x4_splat(r11)));
  nr1 = wasm_f32x4_add(nr1, wasm_f32x4_mul(row2, wasm_f32x4_splat(r12)));

  // Row 2
  v128_t nr2 = wasm_f32x4_mul(row0, wasm_f32x4_splat(r20));
  nr2 = wasm_f32x4_add(nr2, wasm_f32x4_mul(row1, wasm_f32x4_splat(r21)));
  nr2 = wasm_f32x4_add(nr2, wasm_f32x4_mul(row2, wasm_f32x4_splat(r22)));

  wcn_mat4_set_row_wasm(&newDst, 0, nr0);
  wcn_mat4_set_row_wasm(&newDst, 1, nr1);
  wcn_mat4_set_row_wasm(&newDst, 2, nr2);
  wcn_mat4_set_row_wasm(&newDst, 3, row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Implementation
  size_t vl = 4; // Assuming operation on 4 floats

  vfloat32m1_t row0 = wcn_mat4_get_row_riscv(&m, 0, vl);
  vfloat32m1_t row1 = wcn_mat4_get_row_riscv(&m, 1, vl);
  vfloat32m1_t row2 = wcn_mat4_get_row_riscv(&m, 2, vl);
  vfloat32m1_t row3 = wcn_mat4_get_row_riscv(&m, 3, vl);

  // Row 0
  // nr0 = row0 * r00
  vfloat32m1_t nr0 = __riscv_vfmul_vf_f32m1(row0, r00, vl);
  // nr0 += row1 * r01
  nr0 = __riscv_vfmacc_vf_f32m1(nr0, r01, row1, vl);
  // nr0 += row2 * r02
  nr0 = __riscv_vfmacc_vf_f32m1(nr0, r02, row2, vl);

  // Row 1
  vfloat32m1_t nr1 = __riscv_vfmul_vf_f32m1(row0, r10, vl);
  nr1 = __riscv_vfmacc_vf_f32m1(nr1, r11, row1, vl);
  nr1 = __riscv_vfmacc_vf_f32m1(nr1, r12, row2, vl);

  // Row 2
  vfloat32m1_t nr2 = __riscv_vfmul_vf_f32m1(row0, r20, vl);
  nr2 = __riscv_vfmacc_vf_f32m1(nr2, r21, row1, vl);
  nr2 = __riscv_vfmacc_vf_f32m1(nr2, r22, row2, vl);

  wcn_mat4_set_row_riscv(&newDst, 0, nr0, vl);
  wcn_mat4_set_row_riscv(&newDst, 1, nr1, vl);
  wcn_mat4_set_row_riscv(&newDst, 2, nr2, vl);
  wcn_mat4_set_row_riscv(&newDst, 3, row3, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Implementation

  __m128 row0 = wcn_mat4_get_row_loongarch(&m, 0);
  __m128 row1 = wcn_mat4_get_row_loongarch(&m, 1);
  __m128 row2 = wcn_mat4_get_row_loongarch(&m, 2);
  __m128 row3 = wcn_mat4_get_row_loongarch(&m, 3);

  // Row 0
  __m128 v_r00 = __lsx_vreplfr2vr_s(r00);
  __m128 v_r01 = __lsx_vreplfr2vr_s(r01);
  __m128 v_r02 = __lsx_vreplfr2vr_s(r02);

  __m128 nr0 = __lsx_vfmul_s(row0, v_r00);
  nr0 = __lsx_vfmadd_s(row1, v_r01, nr0); // d = a*b + c
  nr0 = __lsx_vfmadd_s(row2, v_r02, nr0);

  // Row 1
  __m128 v_r10 = __lsx_vreplfr2vr_s(r10);
  __m128 v_r11 = __lsx_vreplfr2vr_s(r11);
  __m128 v_r12 = __lsx_vreplfr2vr_s(r12);

  __m128 nr1 = __lsx_vfmul_s(row0, v_r10);
  nr1 = __lsx_vfmadd_s(row1, v_r11, nr1);
  nr1 = __lsx_vfmadd_s(row2, v_r12, nr1);

  // Row 2
  __m128 v_r20 = __lsx_vreplfr2vr_s(r20);
  __m128 v_r21 = __lsx_vreplfr2vr_s(r21);
  __m128 v_r22 = __lsx_vreplfr2vr_s(r22);

  __m128 nr2 = __lsx_vfmul_s(row0, v_r20);
  nr2 = __lsx_vfmadd_s(row1, v_r21, nr2);
  nr2 = __lsx_vfmadd_s(row2, v_r22, nr2);

  wcn_mat4_set_row_loongarch(&newDst, 0, nr0);
  wcn_mat4_set_row_loongarch(&newDst, 1, nr1);
  wcn_mat4_set_row_loongarch(&newDst, 2, nr2);
  wcn_mat4_set_row_loongarch(&newDst, 3, row3);

#else
  // Scalar Fallback

  // Cache rows locally to help compiler optimization (aliasing)
  float m00 = m.m[0]; float m01 = m.m[1]; float m02 = m.m[2]; float m03 = m.m[3];
  float m10 = m.m[4]; float m11 = m.m[5]; float m12 = m.m[6]; float m13 = m.m[7];
  float m20 = m.m[8]; float m21 = m.m[9]; float m22 = m.m[10]; float m23 = m.m[11];
  // m30, m31, m32, m33 are just copied

  newDst.m[0] = r00 * m00 + r01 * m10 + r02 * m20;
  newDst.m[1] = r00 * m01 + r01 * m11 + r02 * m21;
  newDst.m[2] = r00 * m02 + r01 * m12 + r02 * m22;
  newDst.m[3] = r00 * m03 + r01 * m13 + r02 * m23;

  newDst.m[4] = r10 * m00 + r11 * m10 + r12 * m20;
  newDst.m[5] = r10 * m01 + r11 * m11 + r12 * m21;
  newDst.m[6] = r10 * m02 + r11 * m12 + r12 * m22;
  newDst.m[7] = r10 * m03 + r11 * m13 + r12 * m23;

  newDst.m[8] = r20 * m00 + r21 * m10 + r22 * m20;
  newDst.m[9] = r20 * m01 + r21 * m11 + r22 * m21;
  newDst.m[10] = r20 * m02 + r21 * m12 + r22 * m22;
  newDst.m[11] = r20 * m03 + r21 * m13 + r22 * m23;

  // Unconditionally copy the last row
  newDst.m[12] = m.m[12];
  newDst.m[13] = m.m[13];
  newDst.m[14] = m.m[14];
  newDst.m[15] = m.m[15];

#endif

  return newDst;
}

// Mat4 axisRotation
/**
 * Creates a 4-by-4 matrix which rotates around the given axis by the given
 * angle.
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns A matrix which rotates angle radians
 *     around the axis.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, axis_rotation)
(WMATH_TYPE(Vec3) axis, float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  float x = axis.v[0];
  float y = axis.v[1];
  float z = axis.v[2];
  float n = sqrtf(x * x + y * y + z * z);
  x /= n;
  y /= n;
  z /= n;
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float oneMinusCosine = 1 - c;

  newDst.m[0] = xx + (1 - xx) * c;
  newDst.m[1] = x * y * oneMinusCosine + z * s;
  newDst.m[2] = x * z * oneMinusCosine - y * s;
  // newDst.m[3] = 0;
  newDst.m[4] = x * y * oneMinusCosine - z * s;
  newDst.m[5] = yy + (1 - yy) * c;
  newDst.m[6] = y * z * oneMinusCosine + x * s;
  // newDst.m[7] = 0;
  newDst.m[8] = x * z * oneMinusCosine + y * s;
  newDst.m[9] = y * z * oneMinusCosine - x * s;
  newDst.m[10] = zz + (1 - zz) * c;
  // newDst.m[11] = 0;
  // newDst.m[12] = 0;
  // newDst.m[13] = 0;
  // newDst.m[14] = 0;
  newDst.m[15] = 1;

  return newDst;
}

// Mat4 camera_aim
/**
 * Computes a 4-by-4 camera aim transformation.
 *
 * This is a matrix which positions an object aiming down negative Z.
 * toward the target.
 *
 * Note: this is the inverse of `lookAt`
 *
 * @param eye - The position of the object.
 * @param target - The position meant to be aimed at.
 * @param up - A vector pointing up.
 * @returns The aim matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, camera_aim)
(
    //
    WMATH_TYPE(Vec3) eye,    // eye: Vec3
    WMATH_TYPE(Vec3) target, // target: Vec3
    WMATH_TYPE(Vec3) up      // up: Vec3
) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  WMATH_TYPE(Vec3) z_axis = WMATH_NORMALIZE(Vec3)(WMATH_SUB(Vec3)(eye, target));
  WMATH_TYPE(Vec3)
  x_axis = WMATH_NORMALIZE(Vec3)(WMATH_CROSS(Vec3)(up, z_axis));
  WMATH_TYPE(Vec3) y_axis = WMATH_CROSS(Vec3)(z_axis, x_axis);

  newDst.m[0] = x_axis.v[0];
  newDst.m[1] = x_axis.v[1];
  newDst.m[2] = x_axis.v[2]; // x
  // newDst.m[3] = 0;
  newDst.m[4] = y_axis.v[0];
  newDst.m[5] = y_axis.v[1];
  newDst.m[6] = y_axis.v[2]; // y
  // newDst.m[7] = 0;
  newDst.m[8] = z_axis.v[0];
  newDst.m[9] = z_axis.v[1];
  newDst.m[10] = z_axis.v[2]; // z
  // newDst.m[11] = 0;
  newDst.m[12] = eye.v[0];
  newDst.m[13] = eye.v[1];
  newDst.m[14] = eye.v[2];
  newDst.m[15] = 1; // eye

  return newDst;
}

// Mat4 frustum
/**
 * Computes a 4-by-4 perspective transformation matrix given the left, right,
 * top, bottom, near and far clipping planes. The arguments define a frustum
 * extending in the negative z direction. The arguments near and far are the
 * distances to the near and far clipping planes. Note that near and far are not
 * z coordinates, but rather they are distances along the negative z-axis. The
 * matrix generated sends the viewing frustum to the unit box. We assume a unit
 * box extending from -1 to 1 in the x and y dimensions and from 0 to 1 in the z
 * dimension.
 * @param left - The x coordinate of the left plane of the box.
 * @param right - The x coordinate of the right plane of the box.
 * @param bottom - The y coordinate of the bottom plane of the box.
 * @param top - The y coordinate of the right plane of the box.
 * @param near - The negative z coordinate of the near plane of the box.
 * @param far - The negative z coordinate of the far plane of the box.
 * @returns The perspective projection matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, frustum)
(const float left, const float right, const float bottom, const float top, const float near, const float far) {
  WMATH_TYPE(Mat4) newDst;

  // 1. Scalar Pre-calculation
  const float inv_dx = 1.0f / (right - left);
  const float inv_dy = 1.0f / (top - bottom);
  const float inv_dz = 1.0f / (near - far); // Standard OpenGL dz usually (near-far) or (far-near) depending on formula signs

  float m00 = 2.0f * near * inv_dx;
  float m11 = 2.0f * near * inv_dy;
  float m20 = (left + right) * inv_dx;
  float m21 = (top + bottom) * inv_dy;

  // Standard OpenGL Perspective (Z maps to [0, 1] or [-1, 1] depending on API, but here likely [0, 1] based on previous code)
  // Assuming [0, 1] clip space (Vulkan/Metal):
  // m22 = far / (near - far)
  // m32 = (near * far) / (near - far)
  float m22 = far * inv_dz;
  float m32 = near * far * inv_dz;

  // Matrix Layout (Column-Major)
  // Col 0: [m00, 0, 0, 0]
  // Col 1: [0, m11, 0, 0]
  // Col 2: [m20, m21, m22, -1]
  // Col 3: [0, 0, m32, 0]

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE Implementation
  __m128 col0 = _mm_setr_ps(m00, 0.0f, 0.0f, 0.0f);
  __m128 col1 = _mm_setr_ps(0.0f, m11, 0.0f, 0.0f);
  __m128 col2 = _mm_setr_ps(m20, m21, m22, -1.0f);
  __m128 col3 = _mm_setr_ps(0.0f, 0.0f, m32, 0.0f);

  _mm_storeu_ps(&newDst.m[0], col0);
  _mm_storeu_ps(&newDst.m[4], col1);
  _mm_storeu_ps(&newDst.m[8], col2);
  _mm_storeu_ps(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Implementation
  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {m20, m21, m22, -1.0f};
  float c3[4] = {0.0f, 0.0f, m32, 0.0f};

  vst1q_f32(&newDst.m[0], vld1q_f32(c0));
  vst1q_f32(&newDst.m[4], vld1q_f32(c1));
  vst1q_f32(&newDst.m[8], vld1q_f32(c2));
  vst1q_f32(&newDst.m[12], vld1q_f32(c3));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD Implementation
  v128_t col0 = wasm_f32x4_make(m00, 0.0f, 0.0f, 0.0f);
  v128_t col1 = wasm_f32x4_make(0.0f, m11, 0.0f, 0.0f);
  v128_t col2 = wasm_f32x4_make(m20, m21, m22, -1.0f);
  v128_t col3 = wasm_f32x4_make(0.0f, 0.0f, m32, 0.0f);

  wasm_v128_store(&newDst.m[0], col0);
  wasm_v128_store(&newDst.m[4], col1);
  wasm_v128_store(&newDst.m[8], col2);
  wasm_v128_store(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Implementation
  float data[16] = {
    m00, 0.0f, 0.0f, 0.0f,
    0.0f, m11, 0.0f, 0.0f,
    m20, m21, m22, -1.0f,
    0.0f, 0.0f, m32, 0.0f
  };
  size_t vl = __riscv_vsetvli(16, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);
  vfloat32m1_t vec = __riscv_vle32_v_f32m1(data, vl);
  __riscv_vse32_v_f32m1(newDst.m, vec, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Implementation
  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {m20, m21, m22, -1.0f};
  float c3[4] = {0.0f, 0.0f, m32, 0.0f};

  __lsx_vst(__lsx_vld(c0, 0), &newDst.m[0], 0);
  __lsx_vst(__lsx_vld(c1, 0), &newDst.m[4], 0);
  __lsx_vst(__lsx_vld(c2, 0), &newDst.m[8], 0);
  __lsx_vst(__lsx_vld(c3, 0), &newDst.m[12], 0);

#else
  // Scalar Fallback
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0] = m00;
  newDst.m[5] = m11;
  newDst.m[8] = m20;
  newDst.m[9] = m21;
  newDst.m[10] = m22;
  newDst.m[11] = -1.0f;
  newDst.m[14] = m32;
#endif

  return newDst;
}

// Mat4 frustumReverseZ
/**
 * Computes a 4-by-4 reverse-z perspective transformation matrix given the left,
 * right, top, bottom, near and far clipping planes. The arguments define a
 * frustum extending in the negative z direction. The arguments near and far are
 * the distances to the near and far clipping planes. Note that near and far are
 * not z coordinates, but rather they are distances along the negative z-axis.
 * The matrix generated sends the viewing frustum to the unit box. We assume a
 * unit box extending from -1 to 1 in the x and y dimensions and from 1 (-near)
 * to 0 (-far) in the z dimension.
 * @param left - The x coordinate of the left plane of the box.
 * @param right - The x coordinate of the right plane of the box.
 * @param bottom - The y coordinate of the bottom plane of the box.
 * @param top - The y coordinate of the right plane of the box.
 * @param near - The negative z coordinate of the near plane of the box.
 * @param far - The negative z coordinate of the far plane of the box.
 * @returns The perspective projection matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, frustum_reverse_z)
(const float left, const float right, const float bottom, const float top, const float near, const float far) {
  WMATH_TYPE(Mat4) newDst;

  // 1. Scalar Pre-calculation
  const float inv_dx = 1.0f / (right - left);
  const float inv_dy = 1.0f / (top - bottom);

  float m00 = 2.0f * near * inv_dx;
  float m11 = 2.0f * near * inv_dy;
  float m20 = (left + right) * inv_dx;
  float m21 = (top + bottom) * inv_dy;
  float m22, m32;

  // Reverse-Z Logic (Z maps to [1, 0])
  // Standard formulation:
  // z_ndc = (m22 * z_eye + m32 * w_eye) / -z_eye
  // At z=-n -> z_ndc = 1. At z=-f -> z_ndc = 0.

  // Use a local variable to check infinity to handle WMATH_OR_ELSE logic if needed
  // assuming 'far' can be INFINITY
  if (isfinite(far)) {
    // Finite Far Plane
    const float range_inv = 1.0f / (far - near);
    m22 = near * range_inv;          // m[10]
    m32 = far * near * range_inv;    // m[14]
  } else {
    // Infinite Far Plane
    m22 = 0.0f;                      // m[10]
    m32 = near;                      // m[14]
  }

  // Matrix Layout (Column-Major)
  // Col 0: [m00, 0, 0, 0]
  // Col 1: [0, m11, 0, 0]
  // Col 2: [m20, m21, m22, -1]
  // Col 3: [0, 0, m32, 0]

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE Implementation
  __m128 col0 = _mm_setr_ps(m00, 0.0f, 0.0f, 0.0f);
  __m128 col1 = _mm_setr_ps(0.0f, m11, 0.0f, 0.0f);
  __m128 col2 = _mm_setr_ps(m20, m21, m22, -1.0f);
  __m128 col3 = _mm_setr_ps(0.0f, 0.0f, m32, 0.0f);

  _mm_storeu_ps(&newDst.m[0], col0);
  _mm_storeu_ps(&newDst.m[4], col1);
  _mm_storeu_ps(&newDst.m[8], col2);
  _mm_storeu_ps(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Implementation
  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {m20, m21, m22, -1.0f};
  float c3[4] = {0.0f, 0.0f, m32, 0.0f};

  vst1q_f32(&newDst.m[0], vld1q_f32(c0));
  vst1q_f32(&newDst.m[4], vld1q_f32(c1));
  vst1q_f32(&newDst.m[8], vld1q_f32(c2));
  vst1q_f32(&newDst.m[12], vld1q_f32(c3));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD Implementation
  v128_t col0 = wasm_f32x4_make(m00, 0.0f, 0.0f, 0.0f);
  v128_t col1 = wasm_f32x4_make(0.0f, m11, 0.0f, 0.0f);
  v128_t col2 = wasm_f32x4_make(m20, m21, m22, -1.0f);
  v128_t col3 = wasm_f32x4_make(0.0f, 0.0f, m32, 0.0f);

  wasm_v128_store(&newDst.m[0], col0);
  wasm_v128_store(&newDst.m[4], col1);
  wasm_v128_store(&newDst.m[8], col2);
  wasm_v128_store(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Implementation
  float data[16] = {
    m00, 0.0f, 0.0f, 0.0f,
    0.0f, m11, 0.0f, 0.0f,
    m20, m21, m22, -1.0f,
    0.0f, 0.0f, m32, 0.0f
  };
  size_t vl = __riscv_vsetvli(16, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);
  vfloat32m1_t vec = __riscv_vle32_v_f32m1(data, vl);
  __riscv_vse32_v_f32m1(newDst.m, vec, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Implementation
  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {m20, m21, m22, -1.0f};
  float c3[4] = {0.0f, 0.0f, m32, 0.0f};

  __lsx_vst(__lsx_vld(c0, 0), &newDst.m[0], 0);
  __lsx_vst(__lsx_vld(c1, 0), &newDst.m[4], 0);
  __lsx_vst(__lsx_vld(c2, 0), &newDst.m[8], 0);
  __lsx_vst(__lsx_vld(c3, 0), &newDst.m[12], 0);

#else
  // Scalar Fallback
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0] = m00;
  newDst.m[5] = m11;
  newDst.m[8] = m20;
  newDst.m[9] = m21;
  newDst.m[10] = m22;
  newDst.m[11] = -1.0f;
  newDst.m[14] = m32;
#endif

  return newDst;
}

// Mat4 get_axis
/**
 * Returns an axis of a 4x4 matrix as a vector with 3 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y, 2 = z;
 * @returns The axis component of m.
 */
WMATH_TYPE(Vec3)
WMATH_CALL(Mat4, get_axis)
(const WMATH_TYPE(Mat4) m, const int axis) {
  WMATH_TYPE(Vec3) newDst = WMATH_ZERO(Vec3)();
  const int off = axis * 4;
  newDst.v[0] = m.m[off + 0];
  newDst.v[1] = m.m[off + 1];
  newDst.v[2] = m.m[off + 2];
  return newDst;
}

// Mat4 set_axis
/**
 * Sets an axis of a 4x4 matrix as a vector with 3 entries
 * @param m - The matrix.
 * @param v - the axis vector
 * @param axis - The axis  0 = x, 1 = y, 2 = z;
 * @returns The matrix with axis set.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, set_axis)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v, int axis) {
  WMATH_TYPE(Mat4) newDst = WMATH_COPY(Mat4)(m);

  int off = axis * 4;
  newDst.m[off + 0] = v.v[0];
  newDst.m[off + 1] = v.v[1];
  newDst.m[off + 2] = v.v[2];
  return newDst;
}

// Mat4 getTranslation
/**
 * Returns the translation component of a 4-by-4 matrix as a vector with 3
 * entries.
 * @param m - The matrix.
 * @returns The translation component of m.
 */
WMATH_TYPE(Vec3)
WMATH_GET_TRANSLATION(Mat4)
(const WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) v;
  v.v[0] = m.m[12];
  v.v[1] = m.m[13];
  v.v[2] = m.m[14];
  return v;
}

// Mat4 setTranslation
WMATH_TYPE(Mat4)
WMATH_SET_TRANSLATION(Mat4)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst = WMATH_IDENTITY(Mat4)();
  if (!WMATH_EQUALS(Mat4)(m, newDst)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[3] = m.m[3];
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
    newDst.m[7] = m.m[7];
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
    newDst.m[11] = m.m[11];
  }
  newDst.m[12] = v.v[0];
  newDst.m[13] = v.v[1];
  newDst.m[14] = v.v[2];
  newDst.m[15] = 1;
  return newDst;
}

// Mat4 translation
/**
 * Creates a 4-by-4 matrix which translates by the given vector v.
 * @param v - The vector by
 *     which to translate.
 * @returns The translation matrix.
 */
// Mat4 translation
WMATH_TYPE(Mat4)
WMATH_TRANSLATION(Mat4)
(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // ========================= SSE implementation =========================
  // Create translation matrix as 4 rows
  __m128 row0 = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
  __m128 row1 = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
  __m128 row2 = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
  __m128 row3 = _mm_setr_ps(v.v[0], v.v[1], v.v[2], 1.0f);

  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);
  _mm_storeu_ps(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // ========================= NEON implementation =========================
  float32x4_t row0 = vdupq_n_f32(0.0f);
  row0 = vsetq_lane_f32(1.0f, row0, 0);

  float32x4_t row1 = vdupq_n_f32(0.0f);
  row1 = vsetq_lane_f32(1.0f, row1, 1);

  float32x4_t row2 = vdupq_n_f32(0.0f);
  row2 = vsetq_lane_f32(1.0f, row2, 2);

  float32x4_t row3 = vdupq_n_f32(0.0f);
  row3 = vsetq_lane_f32(v.v[0], row3, 0);
  row3 = vsetq_lane_f32(v.v[1], row3, 1);
  row3 = vsetq_lane_f32(v.v[2], row3, 2);
  row3 = vsetq_lane_f32(1.0f, row3, 3);

  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);
  vst1q_f32(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // ========================= WASM SIMD implementation =========================
  v128_t row0 = wasm_v128_const(0x00000000, 0x00000000, 0x00000000, 0x3F800000); // [0, 0, 0, 1.0]
  row0 = wasm_f32x4_replace_lane(row0, 0, 1.0f);

  v128_t row1 = wasm_v128_const(0x00000000, 0x00000000, 0x00000000, 0x3F800000); // [0, 0, 0, 1.0]
  row1 = wasm_f32x4_replace_lane(row1, 1, 1.0f);

  v128_t row2 = wasm_v128_const(0x00000000, 0x00000000, 0x00000000, 0x3F800000); // [0, 0, 0, 1.0]
  row2 = wasm_f32x4_replace_lane(row2, 2, 1.0f);

  v128_t row3 = wasm_v128_const(0x3F800000, 0x00000000, 0x00000000, 0x3F800000); // [1.0, 0, 0, 1.0]
  row3 = wasm_f32x4_replace_lane(row3, 0, v.v[0]);
  row3 = wasm_f32x4_replace_lane(row3, 1, v.v[1]);
  row3 = wasm_f32x4_replace_lane(row3, 2, v.v[2]);

  wasm_v128_store(&newDst.m[0], row0);
  wasm_v128_store(&newDst.m[4], row1);
  wasm_v128_store(&newDst.m[8], row2);
  wasm_v128_store(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // ========================= RISC-V Vector implementation =========================
  size_t vl = vsetvlmax_e32m1();

  // Use scalar approach for translation matrix construction
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0]  = 1.0f;
  newDst.m[5]  = 1.0f;
  newDst.m[10] = 1.0f;
  newDst.m[12] = v.v[0];
  newDst.m[13] = v.v[1];
  newDst.m[14] = v.v[2];
  newDst.m[15] = 1.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // ========================= LoongArch LSX implementation =========================
  __m128 row0 = __lsx_vldrepl_w(&(float){0.0f}, 0);
  row0 = __lsx_vinsgr2vr_w(row0, 1, 0);  // Insert 1.0 at index 0

  __m128 row1 = __lsx_vldrepl_w(&(float){0.0f}, 0);
  row1 = __lsx_vinsgr2vr_w(row1, 1, 1);  // Insert 1.0 at index 1

  __m128 row2 = __lsx_vldrepl_w(&(float){0.0f}, 0);
  row2 = __lsx_vinsgr2vr_w(row2, 1, 2);  // Insert 1.0 at index 2

  __m128 row3 = __lsx_vldrepl_w(&(float){1.0f}, 0);
  row3 = __lsx_vinsgr2vr_w(row3, v.v[0], 0);  // Insert v[0] at index 0
  row3 = __lsx_vinsgr2vr_w(row3, v.v[1], 1);  // Insert v[1] at index 1
  row3 = __lsx_vinsgr2vr_w(row3, v.v[2], 2);  // Insert v[2] at index 2

  __lsx_vst(row0, &newDst.m[0], 0);
  __lsx_vst(row1, &newDst.m[4], 0);
  __lsx_vst(row2, &newDst.m[8], 0);
  __lsx_vst(row3, &newDst.m[12], 0);

#else
  // ========================= Scalar fallback =========================
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0]  = 1.0f;
  newDst.m[5]  = 1.0f;
  newDst.m[10] = 1.0f;
  newDst.m[12] = v.v[0];
  newDst.m[13] = v.v[1];
  newDst.m[14] = v.v[2];
  newDst.m[15] = 1.0f;
#endif

  return newDst;
}


// Mat4 perspective
/**
 * Computes a 4-by-4 perspective transformation matrix given the angular height
 * of the frustum, the aspect ratio, and the near and far clipping planes.  The
 * arguments define a frustum extending in the negative z direction.  The given
 * angle is the vertical angle of the frustum, and the horizontal angle is
 * determined to produce the given aspect ratio.  The arguments near and far are
 * the distances to the near and far clipping planes.  Note that near and far
 * are not z coordinates, but rather they are distances along the negative
 * z-axis.  The matrix generated sends the viewing frustum to the unit box.
 * We assume a unit box extending from -1 to 1 in the x and y dimensions and
 * from 0 to 1 in the z dimension.
 *
 * Note: If you pass `Infinity` for zFar then it will produce a projection
 * matrix returns -Infinity for Z when transforming coordinates with Z <= 0 and
 * +Infinity for Z otherwise.
 *
 * @param fieldOfViewYInRadians - The camera angle from top to bottom (in
 * radians).
 * @param aspect - The aspect ratio width / height.
 * @param zNear - The depth (negative z coordinate)
 *     of the near clipping plane.
 * @param zFar - The depth (negative z coordinate)
 *     of the far clipping plane.
 * @returns The perspective matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, perspective)
(float fieldOfViewYInRadians, float aspect, float zNear, float zFar) {
  WMATH_TYPE(Mat4) newDst;

  // 1. 预先计算标量值 (Scalar Pre-calculation)
  // 透视投影的核心计算通常是标量操作，SIMD化这些操作收益很低。
  // 我们先算出所有非零元素，再用 SIMD 填充。

  float f = 1.0f / tanf(fieldOfViewYInRadians * 0.5f);
  float invAspect = 1.0f / aspect;

  float m00 = f * invAspect;
  float m11 = f;
  float m22, m32;

  // 处理 zFar 无穷大的情况 (Infinite Z)
  // 遵循原始代码的数学逻辑：z映射到 [0, 1]
  if (isfinite(zFar)) {
    float rangeInv = 1.0f / (zNear - zFar);
    m22 = zFar * rangeInv;
    m32 = zFar * zNear * rangeInv;
  } else {
    m22 = -1.0f;
    m32 = -zNear;
  }

  // 注意：newDst.m 内存布局通常是列主序 (Column-Major)
  // Col 0: [m00, 0, 0, 0]
  // Col 1: [0, m11, 0, 0]
  // Col 2: [0, 0, m22, -1]
  // Col 3: [0, 0, m32, 0]

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE Implementation
  // 使用 _mm_setr_ps (set reverse) 直接按内存顺序构建向量

  __m128 col0 = _mm_setr_ps(m00, 0.0f, 0.0f, 0.0f);
  __m128 col1 = _mm_setr_ps(0.0f, m11, 0.0f, 0.0f);
  __m128 col2 = _mm_setr_ps(0.0f, 0.0f, m22, -1.0f);
  __m128 col3 = _mm_setr_ps(0.0f, 0.0f, m32, 0.0f);

  _mm_storeu_ps(&newDst.m[0], col0);
  _mm_storeu_ps(&newDst.m[4], col1);
  _mm_storeu_ps(&newDst.m[8], col2);
  _mm_storeu_ps(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Implementation

  // 构建数组然后加载通常比多条 vsetq_lane 指令流水线效果更好
  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, m22, -1.0f};
  float c3[4] = {0.0f, 0.0f, m32, 0.0f};

  vst1q_f32(&newDst.m[0], vld1q_f32(c0));
  vst1q_f32(&newDst.m[4], vld1q_f32(c1));
  vst1q_f32(&newDst.m[8], vld1q_f32(c2));
  vst1q_f32(&newDst.m[12], vld1q_f32(c3));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD Implementation

  v128_t col0 = wasm_f32x4_make(m00, 0.0f, 0.0f, 0.0f);
  v128_t col1 = wasm_f32x4_make(0.0f, m11, 0.0f, 0.0f);
  v128_t col2 = wasm_f32x4_make(0.0f, 0.0f, m22, -1.0f);
  v128_t col3 = wasm_f32x4_make(0.0f, 0.0f, m32, 0.0f);

  wasm_v128_store(&newDst.m[0], col0);
  wasm_v128_store(&newDst.m[4], col1);
  wasm_v128_store(&newDst.m[8], col2);
  wasm_v128_store(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Implementation
  // 对于构建稀疏向量，直接存储到栈上的数组然后整块向量加载是标准做法

  float data[16] = {
      m00, 0.0f, 0.0f, 0.0f,
      0.0f, m11, 0.0f, 0.0f,
      0.0f, 0.0f, m22, -1.0f,
      0.0f, 0.0f, m32, 0.0f
  };

  size_t vl = __riscv_vsetvli(16, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);
  vfloat32m1_t vec_all = __riscv_vle32_v_f32m1(data, vl);
  __riscv_vse32_v_f32m1(newDst.m, vec_all, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Implementation
  // LSX 没有方便的 "setr" 指令，所以我们用 load

  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, m22, -1.0f};
  float c3[4] = {0.0f, 0.0f, m32, 0.0f};

  __lsx_vst(__lsx_vld(c0, 0), &newDst.m[0], 0);
  __lsx_vst(__lsx_vld(c1, 0), &newDst.m[4], 0);
  __lsx_vst(__lsx_vld(c2, 0), &newDst.m[8], 0);
  __lsx_vst(__lsx_vld(c3, 0), &newDst.m[12], 0);

#else
  // Scalar Fallback
  // Explicitly zero out memory first ensures correct sparse matrix
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));

  newDst.m[0] = m00;
  newDst.m[5] = m11;
  newDst.m[10] = m22;
  newDst.m[11] = -1.0f;
  newDst.m[14] = m32;
#endif

  return newDst;
}

// Mat4 perspective_reverse_z
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, perspective_reverse_z)
(
    //
    float fieldOfViewYInRadians, // fieldOfViewYInRadians: number
    float aspect,                // aspect: number
    float zNear,                 // zNear: number
    float zFar                   // zFar: number
) {
  float _zFar = WMATH_OR_ELSE(zFar, INFINITY);

  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  const float f = 1 / tanf(0.5f * fieldOfViewYInRadians);

  // 0
  newDst.m[0] = f / aspect;
  // 5
  newDst.m[5] = f;
  // 11
  newDst.m[11] = -1;
  if (isfinite(_zFar)) {
    newDst.m[10] = 0;
    newDst.m[14] = zNear;
  } else {
    float rangeInv = 1 / (_zFar - zNear);
    newDst.m[10] = zNear * rangeInv;
    newDst.m[14] = _zFar * zNear * rangeInv;
  }

  return newDst;
}

// Mat4 translate
/**
 * Translates the given 4-by-4 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by
 *     which to translate.
 * @returns The translated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, translate)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized matrix translation
  __m128 vec_v = wcn_load_vec3_partial(v.v);
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Copy the first three rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 0, row0);
    wcn_mat4_set_row(&newDst, 1, row1);
    wcn_mat4_set_row(&newDst, 2, row2);
  }

  // Calculate translation components using SIMD
  // Extract x, y, z components from translation vector
  __m128 v_x = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 v_y = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 v_z = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(2, 2, 2, 2));

  // Calculate dot products for each translation component
  __m128 dot_x = _mm_mul_ps(row0, v_x);
  __m128 dot_y = _mm_mul_ps(row1, v_y);
  __m128 dot_z = _mm_mul_ps(row2, v_z);

  // Sum the dot products and add original translation
  __m128 sum_xy = _mm_add_ps(dot_x, dot_y);
  __m128 sum_xyz = _mm_add_ps(sum_xy, dot_z);
  __m128 trans_sum = _mm_add_ps(sum_xyz, row3);

  // Store the translation row
  wcn_mat4_set_row(&newDst, 3, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized matrix translation
  float32x4_t vec_v = wcn_load_vec3_partial(v.v);
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Copy the first three rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 0, row0);
    wcn_mat4_set_row(&newDst, 1, row1);
    wcn_mat4_set_row(&newDst, 2, row2);
  }

  // Calculate translation components using SIMD
  // Extract x, y, z components from translation vector
  float32x4_t v_x = vdupq_lane_f32(vget_low_f32(vec_v), 0);
  float32x4_t v_y = vdupq_lane_f32(vget_low_f32(vec_v), 1);
  float32x4_t v_z = vdupq_lane_f32(vget_high_f32(vec_v), 0);

  // Calculate dot products for each translation component
  float32x4_t dot_x = vmulq_f32(row0, v_x);
  float32x4_t dot_y = vmulq_f32(row1, v_y);
  float32x4_t dot_z = vmulq_f32(row2, v_z);

  // Sum the dot products and add original translation
  float32x4_t sum_xy = vaddq_f32(dot_x, dot_y);
  float32x4_t sum_xyz = vaddq_f32(sum_xy, dot_z);
  float32x4_t trans_sum = vaddq_f32(sum_xyz, row3);

  // Store the translation row
  wcn_mat4_set_row(&newDst, 3, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - optimized matrix translation
  v128_t vec_v = wcn_load_vec3_partial(v.v);
  v128_t row0 = wcn_mat4_get_row_wasm(&m, 0);
  v128_t row1 = wcn_mat4_get_row_wasm(&m, 1);
  v128_t row2 = wcn_mat4_get_row_wasm(&m, 2);
  v128_t row3 = wcn_mat4_get_row_wasm(&m, 3);

  // Copy the first three rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_wasm(&newDst, 0, row0);
    wcn_mat4_set_row_wasm(&newDst, 1, row1);
    wcn_mat4_set_row_wasm(&newDst, 2, row2);
  }

  // Calculate translation components using SIMD
  // Extract x, y, z components from translation vector
  v128_t v_x = wasm_i32x4_shuffle(vec_v, vec_v, 0, 0, 0, 0);
  v128_t v_y = wasm_i32x4_shuffle(vec_v, vec_v, 1, 1, 1, 1);
  v128_t v_z = wasm_i32x4_shuffle(vec_v, vec_v, 2, 2, 2, 2);

  // Calculate dot products for each translation component
  v128_t dot_x = wasm_f32x4_mul(row0, v_x);
  v128_t dot_y = wasm_f32x4_mul(row1, v_y);
  v128_t dot_z = wasm_f32x4_mul(row2, v_z);

  // Sum the dot products and add original translation
  v128_t sum_xy = wasm_f32x4_add(dot_x, dot_y);
  v128_t sum_xyz = wasm_f32x4_add(sum_xy, dot_z);
  v128_t trans_sum = wasm_f32x4_add(sum_xyz, row3);

  // Store the translation row
  wcn_mat4_set_row_wasm(&newDst, 3, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation - optimized matrix translation
  size_t vl = vsetvlmax_e32m1();

  vfloat32m1_t vec_v = wcn_load_vec3_partial(v.v);
  vfloat32m1_t row0 = wcn_mat4_get_row_riscv(&m, 0, vl);
  vfloat32m1_t row1 = wcn_mat4_get_row_riscv(&m, 1, vl);
  vfloat32m1_t row2 = wcn_mat4_get_row_riscv(&m, 2, vl);
  vfloat32m1_t row3 = wcn_mat4_get_row_riscv(&m, 3, vl);

  // Copy the first three rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_riscv(&newDst, 0, row0, vl);
    wcn_mat4_set_row_riscv(&newDst, 1, row1, vl);
    wcn_mat4_set_row_riscv(&newDst, 2, row2, vl);
  }

  // Extract x, y, z components from translation vector
  float x = vfmv_f_s_f32m1_f32m1(vec_v);
  float y = vfmv_f_s_f32m1_f32m1(vslide1down_vx_f32m1(vec_v, 0, vl));
  float z = vfmv_f_s_f32m1_f32m1(vslide1down_vx_f32m1(vslide1down_vx_f32m1(vec_v, 0, vl), 0, vl));

  vfloat32m1_t v_x = vfmv_v_f_f32m1(x, vl);
  vfloat32m1_t v_y = vfmv_v_f_f32m1(y, vl);
  vfloat32m1_t v_z = vfmv_v_f_f32m1(z, vl);

  // Calculate dot products for each translation component
  vfloat32m1_t dot_x = vfmul_vv_f32m1(row0, v_x, vl);
  vfloat32m1_t dot_y = vfmul_vv_f32m1(row1, v_y, vl);
  vfloat32m1_t dot_z = vfmul_vv_f32m1(row2, v_z, vl);

  // Sum the dot products and add original translation
  vfloat32m1_t sum_xy = vfadd_vv_f32m1(dot_x, dot_y, vl);
  vfloat32m1_t sum_xyz = vfadd_vv_f32m1(sum_xy, dot_z, vl);
  vfloat32m1_t trans_sum = vfadd_vv_f32m1(sum_xyz, row3, vl);

  // Store the translation row
  wcn_mat4_set_row_riscv(&newDst, 3, trans_sum, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - optimized matrix translation
  __m128 vec_v = wcn_load_vec3_partial(v.v);
  __m128 row0 = wcn_mat4_get_row_loongarch(&m, 0);
  __m128 row1 = wcn_mat4_get_row_loongarch(&m, 1);
  __m128 row2 = wcn_mat4_get_row_loongarch(&m, 2);
  __m128 row3 = wcn_mat4_get_row_loongarch(&m, 3);

  // Copy the first three rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_loongarch(&newDst, 0, row0);
    wcn_mat4_set_row_loongarch(&newDst, 1, row1);
    wcn_mat4_set_row_loongarch(&newDst, 2, row2);
  }

  // Calculate translation components using SIMD
  // Extract x, y, z components from translation vector
  __m128 v_x = __lsx_vpickve2gr_w(vec_v, 0);
  v_x = __lsx_vreplvei_w(v_x, 0);
  __m128 v_y = __lsx_vpickve2gr_w(vec_v, 1);
  v_y = __lsx_vreplvei_w(v_y, 0);
  __m128 v_z = __lsx_vpickve2gr_w(vec_v, 2);
  v_z = __lsx_vreplvei_w(v_z, 0);

  // Calculate dot products for each translation component
  __m128 dot_x = __lsx_fmul_s(row0, v_x);
  __m128 dot_y = __lsx_fmul_s(row1, v_y);
  __m128 dot_z = __lsx_fmul_s(row2, v_z);

  // Sum the dot products and add original translation
  __m128 sum_xy = __lsx_fadd_s(dot_x, dot_y);
  __m128 sum_xyz = __lsx_fadd_s(sum_xy, dot_z);
  __m128 trans_sum = __lsx_fadd_s(sum_xyz, row3);

  // Store the translation row
  wcn_mat4_set_row_loongarch(&newDst, 3, trans_sum);

#else
  // Scalar fallback with optimized variable usage
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];

  // Copy rotation/scaling part if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[3] = m.m[3];
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
    newDst.m[7] = m.m[7];
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
    newDst.m[11] = m.m[11];
  }

  // Calculate translation components with optimized ordering
  newDst.m[12] = m.m[0] * v0 + m.m[4] * v1 + m.m[8] * v2 + m.m[12];
  newDst.m[13] = m.m[1] * v0 + m.m[5] * v1 + m.m[9] * v2 + m.m[13];
  newDst.m[14] = m.m[2] * v0 + m.m[6] * v1 + m.m[10] * v2 + m.m[14];
  newDst.m[15] = m.m[3] * v0 + m.m[7] * v1 + m.m[11] * v2 + m.m[15];
#endif

  return newDst;
}

// Mat4 rotate
/**
 * Rotates the given 4-by-4 matrix around the given axis by the
 * given angle. (same as rotate)
 * @param m - The matrix.
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,    // m: Mat4
    WMATH_TYPE(Vec3) axis, // axis: Vec3
    float angleInRadians   // angleInRadians: number
) {
  return WMATH_CALL(Mat4, axis_rotate)(m, axis, angleInRadians);
}

// Mat4 rotate_x
/**
 * Rotates the given 4-by-4 matrix around the x-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE_X(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,  // m: Mat4
    float angleInRadians // angleInRadians: number
) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized X-axis rotation
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = _mm_set1_ps(c);
  __m128 vec_s = _mm_set1_ps(s);

  // For X-axis rotation: row1' = c*row1 + s*row2, row2' = c*row2 - s*row1
#if defined(WCN_HAS_FMA)
  // Use FMA for better performance
  __m128 new_row1 = _mm_fmadd_ps(vec_c, row1, _mm_mul_ps(vec_s, row2));
  __m128 new_row2 =
      _mm_fmadd_ps(vec_c, row2, _mm_mul_ps(_mm_set1_ps(-s), row1));
#else
  __m128 new_row1 =
      _mm_add_ps(_mm_mul_ps(vec_c, row1), _mm_mul_ps(vec_s, row2));
  __m128 new_row2 =
      _mm_sub_ps(_mm_mul_ps(vec_c, row2), _mm_mul_ps(vec_s, row1));
#endif

  // Store results
  wcn_mat4_set_row(&newDst, 0, row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized X-axis rotation
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float32x4_t vec_c = vdupq_n_f32(c);
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_neg_s = vdupq_n_f32(-s);

  // For X-axis rotation: row1' = c*row1 + s*row2, row2' = c*row2 - s*row1
  float32x4_t new_row1 = vmlaq_f32(vmulq_f32(vec_s, row2), vec_c, row1);
  float32x4_t new_row2 = vmlaq_f32(vmulq_f32(vec_neg_s, row1), vec_c, row2);

  // Store results
  wcn_mat4_set_row(&newDst, 0, row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - optimized X-axis rotation
  v128_t row0 = wcn_mat4_get_row_wasm(&m, 0);
  v128_t row1 = wcn_mat4_get_row_wasm(&m, 1);
  v128_t row2 = wcn_mat4_get_row_wasm(&m, 2);
  v128_t row3 = wcn_mat4_get_row_wasm(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  v128_t vec_c = wasm_f32x4_splat(c);
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t vec_neg_s = wasm_f32x4_splat(-s);

  // For X-axis rotation: row1' = c*row1 + s*row2, row2' = c*row2 - s*row1
  v128_t new_row1 = wasm_f32x4_add(wasm_f32x4_mul(vec_c, row1), wasm_f32x4_mul(vec_s, row2));
  v128_t new_row2 = wasm_f32x4_add(wasm_f32x4_mul(vec_c, row2), wasm_f32x4_mul(vec_neg_s, row1));

  // Store results
  wcn_mat4_set_row_wasm(&newDst, 0, row0);
  wcn_mat4_set_row_wasm(&newDst, 1, new_row1);
  wcn_mat4_set_row_wasm(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_wasm(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation - optimized X-axis rotation
  size_t vl = vsetvlmax_e32m1();

  vfloat32m1_t row0 = wcn_mat4_get_row_riscv(&m, 0, vl);
  vfloat32m1_t row1 = wcn_mat4_get_row_riscv(&m, 1, vl);
  vfloat32m1_t row2 = wcn_mat4_get_row_riscv(&m, 2, vl);
  vfloat32m1_t row3 = wcn_mat4_get_row_riscv(&m, 3, vl);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  vfloat32m1_t vec_c = vfmv_v_f_f32m1(c, vl);
  vfloat32m1_t vec_s = vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t vec_neg_s = vfmv_v_f_f32m1(-s, vl);

  // For X-axis rotation: row1' = c*row1 + s*row2, row2' = c*row2 - s*row1
  vfloat32m1_t new_row1 = vfadd_vv_f32m1(vfmul_vv_f32m1(vec_c, row1, vl), vfmul_vv_f32m1(vec_s, row2, vl), vl);
  vfloat32m1_t new_row2 = vfadd_vv_f32m1(vfmul_vv_f32m1(vec_c, row2, vl), vfmul_vv_f32m1(vec_neg_s, row1, vl), vl);

  // Store results
  wcn_mat4_set_row_riscv(&newDst, 0, row0, vl);
  wcn_mat4_set_row_riscv(&newDst, 1, new_row1, vl);
  wcn_mat4_set_row_riscv(&newDst, 2, new_row2, vl);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_riscv(&newDst, 3, row3, vl);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - optimized X-axis rotation
  __m128 row0 = wcn_mat4_get_row_loongarch(&m, 0);
  __m128 row1 = wcn_mat4_get_row_loongarch(&m, 1);
  __m128 row2 = wcn_mat4_get_row_loongarch(&m, 2);
  __m128 row3 = wcn_mat4_get_row_loongarch(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = __lsx_vreplfr2vr_s(c);
  __m128 vec_s = __lsx_vreplfr2vr_s(s);
  __m128 vec_neg_s = __lsx_vreplfr2vr_s(-s);

  // For X-axis rotation: row1' = c*row1 + s*row2, row2' = c*row2 - s*row1
  __m128 new_row1 = __lsx_fadd_s(__lsx_fmul_s(vec_c, row1), __lsx_fmul_s(vec_s, row2));
  __m128 new_row2 = __lsx_fadd_s(__lsx_fmul_s(vec_c, row2), __lsx_fmul_s(vec_neg_s, row1));

  // Store results
  wcn_mat4_set_row_loongarch(&newDst, 0, row0);
  wcn_mat4_set_row_loongarch(&newDst, 1, new_row1);
  wcn_mat4_set_row_loongarch(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_loongarch(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized variable usage
  float m10 = m.m[4];
  float m11 = m.m[5];
  float m12 = m.m[6];
  float m13 = m.m[7];
  float m20 = m.m[8];
  float m21 = m.m[9];
  float m22 = m.m[10];
  float m23 = m.m[11];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[4] = c * m10 + s * m20;
  newDst.m[5] = c * m11 + s * m21;
  newDst.m[6] = c * m12 + s * m22;
  newDst.m[7] = c * m13 + s * m23;
  newDst.m[8] = c * m20 - s * m10;
  newDst.m[9] = c * m21 - s * m11;
  newDst.m[10] = c * m22 - s * m12;
  newDst.m[11] = c * m23 - s * m13;

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[3] = m.m[3];
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 rotate_y
/**
 * Rotates the given 4-by-4 matrix around the y-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE_Y(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,  // m: Mat4
    float angleInRadians // angleInRadians: number
) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized Y-axis rotation
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = _mm_set1_ps(c);
  __m128 vec_s = _mm_set1_ps(s);

  // For Y-axis rotation: row0' = c*row0 - s*row2, row2' = c*row2 + s*row0
#if defined(WCN_HAS_FMA)
  // Use FMA for better performance
  __m128 new_row0 =
      _mm_fmadd_ps(vec_c, row0, _mm_mul_ps(_mm_set1_ps(-s), row2));
  __m128 new_row2 = _mm_fmadd_ps(vec_c, row2, _mm_mul_ps(vec_s, row0));
#else
  __m128 new_row0 =
      _mm_sub_ps(_mm_mul_ps(vec_c, row0), _mm_mul_ps(vec_s, row2));
  __m128 new_row2 =
      _mm_add_ps(_mm_mul_ps(vec_c, row2), _mm_mul_ps(vec_s, row0));
#endif

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized Y-axis rotation
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float32x4_t vec_c = vdupq_n_f32(c);
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_neg_s = vdupq_n_f32(-s);

  // For Y-axis rotation: row0' = c*row0 - s*row2, row2' = c*row2 + s*row0
  float32x4_t new_row0 = vmlaq_f32(vmulq_f32(vec_neg_s, row2), vec_c, row0);
  float32x4_t new_row2 = vmlaq_f32(vmulq_f32(vec_s, row0), vec_c, row2);

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - optimized Y-axis rotation
  v128_t row0 = wcn_mat4_get_row_wasm(&m, 0);
  v128_t row1 = wcn_mat4_get_row_wasm(&m, 1);
  v128_t row2 = wcn_mat4_get_row_wasm(&m, 2);
  v128_t row3 = wcn_mat4_get_row_wasm(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  v128_t vec_c = wasm_f32x4_splat(c);
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t vec_neg_s = wasm_f32x4_splat(-s);

  // For Y-axis rotation: row0' = c*row0 - s*row2, row2' = c*row2 + s*row0
  v128_t new_row0 = wasm_f32x4_sub(wasm_f32x4_mul(vec_c, row0), wasm_f32x4_mul(vec_s, row2));
  v128_t new_row2 = wasm_f32x4_add(wasm_f32x4_mul(vec_c, row2), wasm_f32x4_mul(vec_s, row0));

  // Store results
  wcn_mat4_set_row_wasm(&newDst, 0, new_row0);
  wcn_mat4_set_row_wasm(&newDst, 1, row1);
  wcn_mat4_set_row_wasm(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_wasm(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation - optimized Y-axis rotation
  size_t vl = vsetvlmax_e32m1();

  vfloat32m1_t row0 = wcn_mat4_get_row_riscv(&m, 0, vl);
  vfloat32m1_t row1 = wcn_mat4_get_row_riscv(&m, 1, vl);
  vfloat32m1_t row2 = wcn_mat4_get_row_riscv(&m, 2, vl);
  vfloat32m1_t row3 = wcn_mat4_get_row_riscv(&m, 3, vl);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  vfloat32m1_t vec_c = vfmv_v_f_f32m1(c, vl);
  vfloat32m1_t vec_s = vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t vec_neg_s = vfmv_v_f_f32m1(-s, vl);

  // For Y-axis rotation: row0' = c*row0 - s*row2, row2' = c*row2 + s*row0
  vfloat32m1_t new_row0 = vfsub_vv_f32m1(vfmul_vv_f32m1(vec_c, row0, vl), vfmul_vv_f32m1(vec_s, row2, vl), vl);
  vfloat32m1_t new_row2 = vfadd_vv_f32m1(vfmul_vv_f32m1(vec_c, row2, vl), vfmul_vv_f32m1(vec_s, row0, vl), vl);

  // Store results
  wcn_mat4_set_row_riscv(&newDst, 0, new_row0, vl);
  wcn_mat4_set_row_riscv(&newDst, 1, row1, vl);
  wcn_mat4_set_row_riscv(&newDst, 2, new_row2, vl);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_riscv(&newDst, 3, row3, vl);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - optimized Y-axis rotation
  __m128 row0 = wcn_mat4_get_row_loongarch(&m, 0);
  __m128 row1 = wcn_mat4_get_row_loongarch(&m, 1);
  __m128 row2 = wcn_mat4_get_row_loongarch(&m, 2);
  __m128 row3 = wcn_mat4_get_row_loongarch(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = __lsx_vreplfr2vr_s(c);
  __m128 vec_s = __lsx_vreplfr2vr_s(s);
  __m128 vec_neg_s = __lsx_vreplfr2vr_s(-s);

  // For Y-axis rotation: row0' = c*row0 - s*row2, row2' = c*row2 + s*row0
  __m128 new_row0 = __lsx_fsub_s(__lsx_fmul_s(vec_c, row0), __lsx_fmul_s(vec_s, row2));
  __m128 new_row2 = __lsx_fadd_s(__lsx_fmul_s(vec_c, row2), __lsx_fmul_s(vec_s, row0));

  // Store results
  wcn_mat4_set_row_loongarch(&newDst, 0, new_row0);
  wcn_mat4_set_row_loongarch(&newDst, 1, row1);
  wcn_mat4_set_row_loongarch(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_loongarch(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized variable usage
  float m00 = m.m[0];
  float m01 = m.m[1];
  float m02 = m.m[2];
  float m03 = m.m[3];
  float m20 = m.m[8];
  float m21 = m.m[9];
  float m22 = m.m[10];
  float m23 = m.m[11];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 - s * m20;
  newDst.m[1] = c * m01 - s * m21;
  newDst.m[2] = c * m02 - s * m22;
  newDst.m[3] = c * m03 - s * m23;
  newDst.m[8] = c * m20 + s * m00;
  newDst.m[9] = c * m21 + s * m01;
  newDst.m[10] = c * m22 + s * m02;
  newDst.m[11] = c * m23 + s * m03;

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
    newDst.m[7] = m.m[7];
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 rotate_z
/**
 * Rotates the given 4-by-4 matrix around the z-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE_Z(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,  // m: Mat4
    float angleInRadians // angleInRadians: number
) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized Z-axis rotation
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = _mm_set1_ps(c);
  __m128 vec_s = _mm_set1_ps(s);

  // For Z-axis rotation: row0' = c*row0 + s*row1, row1' = c*row1 - s*row0
#if defined(WCN_HAS_FMA)
  // Use FMA for better performance
  __m128 new_row0 = _mm_fmadd_ps(vec_c, row0, _mm_mul_ps(vec_s, row1));
  __m128 new_row1 =
      _mm_fmadd_ps(vec_c, row1, _mm_mul_ps(_mm_set1_ps(-s), row0));
#else
  __m128 new_row0 =
      _mm_add_ps(_mm_mul_ps(vec_c, row0), _mm_mul_ps(vec_s, row1));
  __m128 new_row1 =
      _mm_sub_ps(_mm_mul_ps(vec_c, row1), _mm_mul_ps(vec_s, row0));
#endif

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized Z-axis rotation
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float32x4_t vec_c = vdupq_n_f32(c);
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_neg_s = vdupq_n_f32(-s);

  // For Z-axis rotation: row0' = c*row0 + s*row1, row1' = c*row1 - s*row0
  float32x4_t new_row0 = vmlaq_f32(vmulq_f32(vec_s, row1), vec_c, row0);
  float32x4_t new_row1 = vmlaq_f32(vmulq_f32(vec_neg_s, row0), vec_c, row1);

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - optimized Z-axis rotation
  v128_t row0 = wcn_mat4_get_row_wasm(&m, 0);
  v128_t row1 = wcn_mat4_get_row_wasm(&m, 1);
  v128_t row2 = wcn_mat4_get_row_wasm(&m, 2);
  v128_t row3 = wcn_mat4_get_row_wasm(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  v128_t vec_c = wasm_f32x4_splat(c);
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t vec_neg_s = wasm_f32x4_splat(-s);

  // For Z-axis rotation: row0' = c*row0 + s*row1, row1' = c*row1 - s*row0
  v128_t new_row0 = wasm_f32x4_add(wasm_f32x4_mul(vec_c, row0), wasm_f32x4_mul(vec_s, row1));
  v128_t new_row1 = wasm_f32x4_add(wasm_f32x4_mul(vec_c, row1), wasm_f32x4_mul(vec_neg_s, row0));

  // Store results
  wcn_mat4_set_row_wasm(&newDst, 0, new_row0);
  wcn_mat4_set_row_wasm(&newDst, 1, new_row1);
  wcn_mat4_set_row_wasm(&newDst, 2, row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_wasm(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation - optimized Z-axis rotation
  size_t vl = vsetvlmax_e32m1();

  vfloat32m1_t row0 = wcn_mat4_get_row_riscv(&m, 0, vl);
  vfloat32m1_t row1 = wcn_mat4_get_row_riscv(&m, 1, vl);
  vfloat32m1_t row2 = wcn_mat4_get_row_riscv(&m, 2, vl);
  vfloat32m1_t row3 = wcn_mat4_get_row_riscv(&m, 3, vl);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  vfloat32m1_t vec_c = vfmv_v_f_f32m1(c, vl);
  vfloat32m1_t vec_s = vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t vec_neg_s = vfmv_v_f_f32m1(-s, vl);

  // For Z-axis rotation: row0' = c*row0 + s*row1, row1' = c*row1 - s*row0
  vfloat32m1_t new_row0 = vfadd_vv_f32m1(vfmul_vv_f32m1(vec_c, row0, vl), vfmul_vv_f32m1(vec_s, row1, vl), vl);
  vfloat32m1_t new_row1 = vfadd_vv_f32m1(vfmul_vv_f32m1(vec_c, row1, vl), vfmul_vv_f32m1(vec_neg_s, row0, vl), vl);

  // Store results
  wcn_mat4_set_row_riscv(&newDst, 0, new_row0, vl);
  wcn_mat4_set_row_riscv(&newDst, 1, new_row1, vl);
  wcn_mat4_set_row_riscv(&newDst, 2, row2, vl);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_riscv(&newDst, 3, row3, vl);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - optimized Z-axis rotation
  __m128 row0 = wcn_mat4_get_row_loongarch(&m, 0);
  __m128 row1 = wcn_mat4_get_row_loongarch(&m, 1);
  __m128 row2 = wcn_mat4_get_row_loongarch(&m, 2);
  __m128 row3 = wcn_mat4_get_row_loongarch(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = __lsx_vreplfr2vr_s(c);
  __m128 vec_s = __lsx_vreplfr2vr_s(s);
  __m128 vec_neg_s = __lsx_vreplfr2vr_s(-s);

  // For Z-axis rotation: row0' = c*row0 + s*row1, row1' = c*row1 - s*row0
  __m128 new_row0 = __lsx_fadd_s(__lsx_fmul_s(vec_c, row0), __lsx_fmul_s(vec_s, row1));
  __m128 new_row1 = __lsx_fadd_s(__lsx_fmul_s(vec_c, row1), __lsx_fmul_s(vec_neg_s, row0));

  // Store results
  wcn_mat4_set_row_loongarch(&newDst, 0, new_row0);
  wcn_mat4_set_row_loongarch(&newDst, 1, new_row1);
  wcn_mat4_set_row_loongarch(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row_loongarch(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized variable usage
  float m00 = m.m[0];
  float m01 = m.m[1];
  float m02 = m.m[2];
  float m03 = m.m[3];
  float m10 = m.m[4];
  float m11 = m.m[5];
  float m12 = m.m[6];
  float m13 = m.m[7];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 + s * m10;
  newDst.m[1] = c * m01 + s * m11;
  newDst.m[2] = c * m02 + s * m12;
  newDst.m[3] = c * m03 + s * m13;
  newDst.m[4] = c * m10 - s * m00;
  newDst.m[5] = c * m11 - s * m01;
  newDst.m[6] = c * m12 - s * m02;
  newDst.m[7] = c * m13 - s * m03;

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
    newDst.m[11] = m.m[11];
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 rotation
/**
 * Creates a 4-by-4 matrix which rotates around the given axis by the given
 * angle. (same as axisRotation)
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns A matrix which rotates angle radians
 *     around the axis.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION(Mat4)
(WMATH_TYPE(Vec3) axis, float angleInRadians) {
  return WMATH_CALL(Mat4, axis_rotation)(axis, angleInRadians);
}

// Mat4 rotation_x
/**
 * Creates a 4-by-4 matrix which rotates around the x-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_X(Mat4)
(float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = 1;
  newDst.m[5] = c;
  newDst.m[6] = s;
  newDst.m[9] = -s;
  newDst.m[10] = c;
  newDst.m[15] = 1;

  return newDst;
}

// Mat4 rotation_y
/**
 * Creates a 4-by-4 matrix which rotates around the y-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_Y(Mat4)
(float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c;
  newDst.m[2] = -s;
  newDst.m[5] = 1;
  newDst.m[8] = s;
  newDst.m[10] = c;
  newDst.m[15] = 1;

  return newDst;
}

// Mat4 rotation_z
/**
 * Creates a 4-by-4 matrix which rotates around the z-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_Z(Mat4)
(float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c;
  newDst.m[1] = s;
  newDst.m[4] = -s;
  newDst.m[5] = c;
  newDst.m[10] = 1;
  newDst.m[15] = 1;

  return newDst;
}

// All Type Scale Impl
WMATH_TYPE(Vec2)
WMATH_SCALE(Vec2)
(WMATH_TYPE(Vec2) v, float scale) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_scale = _mm_set1_ps(scale);
  __m128 vec_res = _mm_mul_ps(vec_v, vec_scale);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_scale = vdupq_n_f32(scale);
  float32x4_t vec_res = vmulq_f32(vec_v, vec_scale);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_v = wcn_load_vec2_partial_wasm(v.v);
  v128_t vec_scale = wasm_f32x4_splat(scale);
  v128_t vec_res = wasm_f32x4_mul(vec_v, vec_scale);
  wcn_store_vec2_partial_wasm(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation
  size_t vl = __riscv_vsetvli(4, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);
  vfloat32m1_t vec_v = wcn_load_vec2_partial_riscv(v.v, vl);
  vfloat32m1_t vec_scale = __riscv_vfmv_v_f_f32m1(scale, vl);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_v, vec_scale, vl);
  wcn_store_vec2_partial_riscv(result.v, vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v = wcn_load_vec2_partial_loongarch(v.v);
  __m128 vec_scale = __lsx_vreplfr2vr_s(scale);
  __m128 vec_res = __lsx_vfmul_s(vec_v, vec_scale);
  wcn_store_vec2_partial_loongarch(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = v.v[0] * scale;
  result.v[1] = v.v[1] * scale;
#endif

  return result;
}

WMATH_TYPE(Vec3)
WMATH_SCALE(Vec3)
(WMATH_TYPE(Vec3) v, float scale) {
  return WMATH_MULTIPLY_SCALAR(Vec3)(v, scale);
}

WMATH_TYPE(Quat)
WMATH_SCALE(Quat)
(WMATH_TYPE(Quat) q, float scale) {
  return WMATH_MULTIPLY_SCALAR(Quat)(q, scale);
}

WMATH_TYPE(Mat3)
WMATH_SCALE(Mat3)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_v0 = _mm_set1_ps(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = _mm_set1_ps(v.v[1]); // Broadcast v1 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = _mm_loadu_ps(&m.m[0]);
  __m128 vec_result0 = _mm_mul_ps(vec_row0, vec_v0);
  _mm_storeu_ps(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = _mm_loadu_ps(&m.m[4]);
  __m128 vec_result1 = _mm_mul_ps(vec_row1, vec_v1);
  _mm_storeu_ps(&newDst.m[4], vec_result1);

  // Copy third row (indices 8-11) if needed
  if (_mm_movemask_ps(_mm_cmpneq_ps(vec_result0, vec_row0)) != 0 ||
      _mm_movemask_ps(_mm_cmpneq_ps(vec_result1, vec_row1)) != 0) {
    __m128 vec_row2 = _mm_loadu_ps(&m.m[8]);
    _mm_storeu_ps(&newDst.m[8], vec_row2);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_v0 = vdupq_n_f32(v.v[0]); // Broadcast v0 to all elements
  float32x4_t vec_v1 = vdupq_n_f32(v.v[1]); // Broadcast v1 to all elements

  // Scale first row (indices 0-3)
  float32x4_t vec_row0 = vld1q_f32(&m.m[0]);
  float32x4_t vec_result0 = vmulq_f32(vec_row0, vec_v0);
  vst1q_f32(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  float32x4_t vec_row1 = vld1q_f32(&m.m[4]);
  float32x4_t vec_result1 = vmulq_f32(vec_row1, vec_v1);
  vst1q_f32(&newDst.m[4], vec_result1);

  // Copy third row (indices 8-11) if needed
  uint32x4_t cmp0 = vceqq_f32(vec_result0, vec_row0);
  uint32x4_t cmp1 = vceqq_f32(vec_result1, vec_row1);
  if (vminvq_u32(cmp0) == 0 || vminvq_u32(cmp1) == 0) {
    float32x4_t vec_row2 = vld1q_f32(&m.m[8]);
    vst1q_f32(&newDst.m[8], vec_row2);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_v0 = wasm_f32x4_splat(v.v[0]); // Broadcast v0 to all elements
  v128_t vec_v1 = wasm_f32x4_splat(v.v[1]); // Broadcast v1 to all elements

  // Scale first row (indices 0-3)
  v128_t vec_row0 = wasm_v128_load(&m.m[0]);
  v128_t vec_result0 = wasm_f32x4_mul(vec_row0, vec_v0);
  wasm_v128_store(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  v128_t vec_row1 = wasm_v128_load(&m.m[4]);
  v128_t vec_result1 = wasm_f32x4_mul(vec_row1, vec_v1);
  wasm_v128_store(&newDst.m[4], vec_result1);

  // Copy third row (indices 8-11) if needed
  int cmp0 = wasm_f32x4_extract_lane(vec_result0, 0) != wasm_f32x4_extract_lane(vec_row0, 0) ||
             wasm_f32x4_extract_lane(vec_result0, 1) != wasm_f32x4_extract_lane(vec_row0, 1) ||
             wasm_f32x4_extract_lane(vec_result0, 2) != wasm_f32x4_extract_lane(vec_row0, 2) ||
             wasm_f32x4_extract_lane(vec_result0, 3) != wasm_f32x4_extract_lane(vec_row0, 3);
  int cmp1 = wasm_f32x4_extract_lane(vec_result1, 0) != wasm_f32x4_extract_lane(vec_row1, 0) ||
             wasm_f32x4_extract_lane(vec_result1, 1) != wasm_f32x4_extract_lane(vec_row1, 1) ||
             wasm_f32x4_extract_lane(vec_result1, 2) != wasm_f32x4_extract_lane(vec_row1, 2) ||
             wasm_f32x4_extract_lane(vec_result1, 3) != wasm_f32x4_extract_lane(vec_row1, 3);
  if (cmp0 || cmp1) {
    wasm_v128_store(&newDst.m[8], wasm_v128_load(&m.m[8]));  // Copy the third row
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector implementation
  size_t vl = __riscv_vsetvli(4, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);

  vfloat32m1_t vec_v0 = __riscv_vfmv_v_f_f32m1(v.v[0], vl); // Broadcast v0 to all elements
  vfloat32m1_t vec_v1 = __riscv_vfmv_v_f_f32m1(v.v[1], vl); // Broadcast v1 to all elements

  // Scale first row (indices 0-3)
  vfloat32m1_t vec_row0 = __riscv_vle32_v_f32m1(&m.m[0], vl);
  vfloat32m1_t vec_result0 = __riscv_vfmul_vv_f32m1(vec_row0, vec_v0, vl);
  __riscv_vse32_v_f32m1(&newDst.m[0], vec_result0, vl);

  // Scale second row (indices 4-7)
  vfloat32m1_t vec_row1 = __riscv_vle32_v_f32m1(&m.m[4], vl);
  vfloat32m1_t vec_result1 = __riscv_vfmul_vv_f32m1(vec_row1, vec_v1, vl);
  __riscv_vse32_v_f32m1(&newDst.m[4], vec_result1, vl);

  // Copy third row (indices 8-11) if needed
  vbool32_t cmp0 = __riscv_vmfeq_vv_f32m1_b32(vec_result0, vec_row0, vl);
  vbool32_t cmp1 = __riscv_vmfeq_vv_f32m1_b32(vec_result1, vec_row1, vl);
  if (__riscv_vfirst_m_b32(__riscv_vmnot_m_b32(cmp0, vl), vl) >= 0 ||
      __riscv_vfirst_m_b32(__riscv_vmnot_m_b32(cmp1, vl), vl) >= 0) {
    vfloat32m1_t vec_row2 = __riscv_vle32_v_f32m1(&m.m[8], vl);
    __riscv_vse32_v_f32m1(&newDst.m[8], vec_row2, vl);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v0 = __lsx_vreplfr2vr_s(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = __lsx_vreplfr2vr_s(v.v[1]); // Broadcast v1 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = __lsx_vld(&m.m[0], 0);
  __m128 vec_result0 = __lsx_vfmul_s(vec_row0, vec_v0);
  __lsx_vst(vec_result0, &newDst.m[0], 0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = __lsx_vld(&m.m[4], 0);
  __m128 vec_result1 = __lsx_vfmul_s(vec_row1, vec_v1);
  __lsx_vst(vec_result1, &newDst.m[4], 0);

  // Copy third row (indices 8-11) if needed
  int cmp0 = __lsx_vpickve2gr_w(vec_result0, 0) != __lsx_vpickve2gr_w(vec_row0, 0) ||
             __lsx_vpickve2gr_w(vec_result0, 1) != __lsx_vpickve2gr_w(vec_row0, 1) ||
             __lsx_vpickve2gr_w(vec_result0, 2) != __lsx_vpickve2gr_w(vec_row0, 2) ||
             __lsx_vpickve2gr_w(vec_result0, 3) != __lsx_vpickve2gr_w(vec_row0, 3);
  int cmp1 = __lsx_vpickve2gr_w(vec_result1, 0) != __lsx_vpickve2gr_w(vec_row1, 0) ||
             __lsx_vpickve2gr_w(vec_result1, 1) != __lsx_vpickve2gr_w(vec_row1, 1) ||
             __lsx_vpickve2gr_w(vec_result1, 2) != __lsx_vpickve2gr_w(vec_row1, 2) ||
             __lsx_vpickve2gr_w(vec_result1, 3) != __lsx_vpickve2gr_w(vec_row1, 3);
  if (cmp0 || cmp1) {
    __m128 vec_row2 = __lsx_vld(&m.m[8], 0);
    __lsx_vst(vec_row2, &newDst.m[8], 0);
  }

#else
  // Scalar fallback (original implementation)
  float v0 = v.v[0];
  float v1 = v.v[1];

  newDst.m[0] = v0 * m.m[0 * 4 + 0];
  newDst.m[1] = v0 * m.m[0 * 4 + 1];
  newDst.m[2] = v0 * m.m[0 * 4 + 2];

  newDst.m[4] = v1 * m.m[1 * 4 + 0];
  newDst.m[5] = v1 * m.m[1 * 4 + 1];
  newDst.m[6] = v1 * m.m[1 * 4 + 2];

  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }
#endif

  return newDst;
}

WMATH_TYPE(Mat4)
WMATH_SCALE(Mat4)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_v0 = _mm_set1_ps(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = _mm_set1_ps(v.v[1]); // Broadcast v1 to all elements
  __m128 vec_v2 = _mm_set1_ps(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = _mm_loadu_ps(&m.m[0]);
  __m128 vec_result0 = _mm_mul_ps(vec_row0, vec_v0);
  _mm_storeu_ps(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = _mm_loadu_ps(&m.m[4]);
  __m128 vec_result1 = _mm_mul_ps(vec_row1, vec_v1);
  _mm_storeu_ps(&newDst.m[4], vec_result1);

  // Scale third row (indices 8-11)
  __m128 vec_row2 = _mm_loadu_ps(&m.m[8]);
  __m128 vec_result2 = _mm_mul_ps(vec_row2, vec_v2);
  _mm_storeu_ps(&newDst.m[8], vec_result2);

  // Check if any element changed and copy last row if needed
  __m128 cmp0 = _mm_cmpneq_ps(vec_result0, vec_row0);
  __m128 cmp1 = _mm_cmpneq_ps(vec_result1, vec_row1);
  __m128 cmp2 = _mm_cmpneq_ps(vec_result2, vec_row2);

  if (_mm_movemask_ps(_mm_or_ps(_mm_or_ps(cmp0, cmp1), cmp2)) != 0) {
    __m128 vec_row3 = _mm_loadu_ps(&m.m[12]);
    _mm_storeu_ps(&newDst.m[12], vec_row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_v0 = vdupq_n_f32(v.v[0]); // Broadcast v0 to all elements
  float32x4_t vec_v1 = vdupq_n_f32(v.v[1]); // Broadcast v1 to all elements
  float32x4_t vec_v2 = vdupq_n_f32(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  float32x4_t vec_row0 = vld1q_f32(&m.m[0]);
  float32x4_t vec_result0 = vmulq_f32(vec_row0, vec_v0);
  vst1q_f32(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  float32x4_t vec_row1 = vld1q_f32(&m.m[4]);
  float32x4_t vec_result1 = vmulq_f32(vec_row1, vec_v1);
  vst1q_f32(&newDst.m[4], vec_result1);

  // Scale third row (indices 8-11)
  float32x4_t vec_row2 = vld1q_f32(&m.m[8]);
  float32x4_t vec_result2 = vmulq_f32(vec_row2, vec_v2);
  vst1q_f32(&newDst.m[8], vec_result2);

  // Check if any element changed and copy last row if needed
  uint32x4_t cmp0 = vceqq_f32(vec_result0, vec_row0);
  uint32x4_t cmp1 = vceqq_f32(vec_result1, vec_row1);
  uint32x4_t cmp2 = vceqq_f32(vec_result2, vec_row2);

  if (vminvq_u32(cmp0) == 0 || vminvq_u32(cmp1) == 0 || vminvq_u32(cmp2) == 0) {
    float32x4_t vec_row3 = vld1q_f32(&m.m[12]);
    vst1q_f32(&newDst.m[12], vec_row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_v0 = wasm_f32x4_splat(v.v[0]); // Broadcast v0 to all elements
  v128_t vec_v1 = wasm_f32x4_splat(v.v[1]); // Broadcast v1 to all elements
  v128_t vec_v2 = wasm_f32x4_splat(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  v128_t vec_row0 = wasm_v128_load(&m.m[0]);
  v128_t vec_result0 = wasm_f32x4_mul(vec_row0, vec_v0);
  wasm_v128_store(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  v128_t vec_row1 = wasm_v128_load(&m.m[4]);
  v128_t vec_result1 = wasm_f32x4_mul(vec_row1, vec_v1);
  wasm_v128_store(&newDst.m[4], vec_result1);

  // Scale third row (indices 8-11)
  v128_t vec_row2 = wasm_v128_load(&m.m[8]);
  v128_t vec_result2 = wasm_f32x4_mul(vec_row2, vec_v2);
  wasm_v128_store(&newDst.m[8], vec_result2);

  // WASM doesn't have the same comparison logic, so we'll check by default
  v128_t vec_row3 = wasm_v128_load(&m.m[12]);
  wasm_v128_store(&newDst.m[12], vec_row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvli(4, __RISCV_VTYPE_F32, __RISCV_VLMUL_1); // Set vector length to 4 for 4 floats

  // Create broadcast vectors for scaling values
  vfloat32m1_t vec_v0 = __riscv_vfmv_v_f_f32m1(v.v[0], vl); // Broadcast v0 to all elements
  vfloat32m1_t vec_v1 = __riscv_vfmv_v_f_f32m1(v.v[1], vl); // Broadcast v1 to all elements
  vfloat32m1_t vec_v2 = __riscv_vfmv_v_f_f32m1(v.v[2], vl); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  vfloat32m1_t vec_row0 = __riscv_vle32_v_f32m1(&m.m[0], vl);  // Load first row
  vfloat32m1_t vec_result0 = __riscv_vfmul_vv_f32m1(vec_row0, vec_v0, vl);  // Multiply by v0
  __riscv_vse32_v_f32m1(&newDst.m[0], vec_result0, vl);  // Store result

  // Scale second row (indices 4-7)
  vfloat32m1_t vec_row1 = __riscv_vle32_v_f32m1(&m.m[4], vl);  // Load second row
  vfloat32m1_t vec_result1 = __riscv_vfmul_vv_f32m1(vec_row1, vec_v1, vl);  // Multiply by v1
  __riscv_vse32_v_f32m1(&newDst.m[4], vec_result1, vl);  // Store result

  // Scale third row (indices 8-11)
  vfloat32m1_t vec_row2 = __riscv_vle32_v_f32m1(&m.m[8], vl);  // Load third row
  vfloat32m1_t vec_result2 = __riscv_vfmul_vv_f32m1(vec_row2, vec_v2, vl);  // Multiply by v2
  __riscv_vse32_v_f32m1(&newDst.m[8], vec_result2, vl);  // Store result

  // Check if any element changed, then copy last row
  vbool32_t cmp0 = __riscv_vmfne_vv_f32m1_b32(vec_result0, vec_row0, vl);
  vbool32_t cmp1 = __riscv_vmfne_vv_f32m1_b32(vec_result1, vec_row1, vl);
  vbool32_t cmp2 = __riscv_vmfne_vv_f32m1_b32(vec_result2, vec_row2, vl);

  // If any element changed in any row, copy the last row
  if (__riscv_vfirst_m_b32(__riscv_vor_mm_b32(__riscv_vor_mm_b32(cmp0, cmp1, vl), cmp2, vl), vl) >= 0) {
    vfloat32m1_t vec_row3 = __riscv_vle32_v_f32m1(&m.m[12], vl);
    __riscv_vse32_v_f32m1(&newDst.m[12], vec_row3, vl);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v0 = __lsx_vldrepl_w(&v.v[0], 0); // Broadcast v0 to all elements
  __m128 vec_v1 = __lsx_vldrepl_w(&v.v[1], 0); // Broadcast v1 to all elements
  __m128 vec_v2 = __lsx_vldrepl_w(&v.v[2], 0); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = __lsx_vld(&m.m[0], 0);
  __m128 vec_result0 = __lsx_vfmul_s(vec_row0, vec_v0);
  __lsx_vst(vec_result0, &newDst.m[0], 0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = __lsx_vld(&m.m[4], 0);
  __m128 vec_result1 = __lsx_vfmul_s(vec_row1, vec_v1);
  __lsx_vst(vec_result1, &newDst.m[4], 0);

  // Scale third row (indices 8-11)
  __m128 vec_row2 = __lsx_vld(&m.m[8], 0);
  __m128 vec_result2 = __lsx_vfmul_s(vec_row2, vec_v2);
  __lsx_vst(vec_result2, &newDst.m[8], 0);

  // Simple check - copy last row
  __m128 vec_row3 = __lsx_vld(&m.m[12], 0);
  __lsx_vst(vec_row3, &newDst.m[12], 0);

#else
  // Scalar fallback (original implementation)
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];

  newDst.m[0] = v0 * m.m[0 * 4 + 0];
  newDst.m[1] = v0 * m.m[0 * 4 + 1];
  newDst.m[2] = v0 * m.m[0 * 4 + 2];
  newDst.m[3] = v0 * m.m[0 * 4 + 3];
  newDst.m[4] = v1 * m.m[1 * 4 + 0];
  newDst.m[5] = v1 * m.m[1 * 4 + 1];
  newDst.m[6] = v1 * m.m[1 * 4 + 2];
  newDst.m[7] = v1 * m.m[1 * 4 + 3];
  newDst.m[8] = v2 * m.m[2 * 4 + 0];
  newDst.m[9] = v2 * m.m[2 * 4 + 1];
  newDst.m[10] = v2 * m.m[2 * 4 + 2];
  newDst.m[11] = v2 * m.m[2 * 4 + 3];

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat3 scale3D
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scale3D)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_v0 = _mm_set1_ps(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = _mm_set1_ps(v.v[1]); // Broadcast v1 to all elements
  __m128 vec_v2 = _mm_set1_ps(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = wcn_mat3_get_row(&m, 0);
  __m128 vec_result0 = _mm_mul_ps(vec_row0, vec_v0);
  wcn_mat3_set_row(&newDst, 0, vec_result0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = wcn_mat3_get_row(&m, 1);
  __m128 vec_result1 = _mm_mul_ps(vec_row1, vec_v1);
  wcn_mat3_set_row(&newDst, 1, vec_result1);

  // Scale third row (indices 8-11)
  __m128 vec_row2 = wcn_mat3_get_row(&m, 2);
  __m128 vec_result2 = _mm_mul_ps(vec_row2, vec_v2);
  wcn_mat3_set_row(&newDst, 2, vec_result2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_v0 = vdupq_n_f32(v.v[0]); // Broadcast v0 to all elements
  float32x4_t vec_v1 = vdupq_n_f32(v.v[1]); // Broadcast v1 to all elements
  float32x4_t vec_v2 = vdupq_n_f32(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  float32x4_t vec_row0 = wcn_mat3_get_row(&m, 0);
  float32x4_t vec_result0 = vmulq_f32(vec_row0, vec_v0);
  wcn_mat3_set_row(&newDst, 0, vec_result0);

  // Scale second row (indices 4-7)
  float32x4_t vec_row1 = wcn_mat3_get_row(&m, 1);
  float32x4_t vec_result1 = vmulq_f32(vec_row1, vec_v1);
  wcn_mat3_set_row(&newDst, 1, vec_result1);

  // Scale third row (indices 8-11)
  float32x4_t vec_row2 = wcn_mat3_get_row(&m, 2);
  float32x4_t vec_result2 = vmulq_f32(vec_row2, vec_v2);
  wcn_mat3_set_row(&newDst, 2, vec_result2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);

  // Broadcast v0, v1, v2 to all elements
  vfloat32m1_t vec_v0 = __riscv_vfmv_v_f_f32m1(v.v[0], vl);
  vfloat32m1_t vec_v1 = __riscv_vfmv_v_f_f32m1(v.v[1], vl);
  vfloat32m1_t vec_v2 = __riscv_vfmv_v_f_f32m1(v.v[2], vl);

  // Scale first row (indices 0-3)
  vfloat32m1_t vec_row0 = wcn_mat3_get_row(&m, 0);
  vfloat32m1_t vec_result0 = __riscv_vfmul_vv_f32m1(vec_row0, vec_v0, vl);
  wcn_mat3_set_row(&newDst, 0, vec_result0);

  // Scale second row (indices 4-7)
  vfloat32m1_t vec_row1 = wcn_mat3_get_row(&m, 1);
  vfloat32m1_t vec_result1 = __riscv_vfmul_vv_f32m1(vec_row1, vec_v1, vl);
  wcn_mat3_set_row(&newDst, 1, vec_result1);

  // Scale third row (indices 8-11)
  vfloat32m1_t vec_row2 = wcn_mat3_get_row(&m, 2);
  vfloat32m1_t vec_result2 = __riscv_vfmul_vv_f32m1(vec_row2, vec_v2, vl);
  wcn_mat3_set_row(&newDst, 2, vec_result2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v0 = __lsx_vreplgr2vr_s(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = __lsx_vreplgr2vr_s(v.v[1]); // Broadcast v1 to all elements
  __m128 vec_v2 = __lsx_vreplgr2vr_s(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = wcn_mat3_get_row(&m, 0);
  __m128 vec_result0 = __lsx_vfmul_s(vec_row0, vec_v0);
  wcn_mat3_set_row(&newDst, 0, vec_result0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = wcn_mat3_get_row(&m, 1);
  __m128 vec_result1 = __lsx_vfmul_s(vec_row1, vec_v1);
  wcn_mat3_set_row(&newDst, 1, vec_result1);

  // Scale third row (indices 8-11)
  __m128 vec_row2 = wcn_mat3_get_row(&m, 2);
  __m128 vec_result2 = __lsx_vfmul_s(vec_row2, vec_v2);
  wcn_mat3_set_row(&newDst, 2, vec_result2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_v0 = wasm_f32x4_splat(v.v[0]); // Broadcast v0 to all elements
  v128_t vec_v1 = wasm_f32x4_splat(v.v[1]); // Broadcast v1 to all elements
  v128_t vec_v2 = wasm_f32x4_splat(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  v128_t vec_row0 = wcn_mat3_get_row(&m, 0);
  v128_t vec_result0 = wasm_f32x4_mul(vec_row0, vec_v0);
  wcn_mat3_set_row(&newDst, 0, vec_result0);

  // Scale second row (indices 4-7)
  v128_t vec_row1 = wcn_mat3_get_row(&m, 1);
  v128_t vec_result1 = wasm_f32x4_mul(vec_row1, vec_v1);
  wcn_mat3_set_row(&newDst, 1, vec_result1);

  // Scale third row (indices 8-11)
  v128_t vec_row2 = wcn_mat3_get_row(&m, 2);
  v128_t vec_result2 = wasm_f32x4_mul(vec_row2, vec_v2);
  wcn_mat3_set_row(&newDst, 2, vec_result2);

#else
  // Scalar fallback (original implementation)
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];
  newDst.m[0] = v0 * m.m[0 * 4 + 0];
  newDst.m[1] = v0 * m.m[0 * 4 + 1];
  newDst.m[2] = v0 * m.m[0 * 4 + 2];
  newDst.m[4] = v1 * m.m[1 * 4 + 0];
  newDst.m[5] = v1 * m.m[1 * 4 + 1];
  newDst.m[6] = v1 * m.m[1 * 4 + 2];
  newDst.m[8] = v2 * m.m[2 * 4 + 0];
  newDst.m[9] = v2 * m.m[2 * 4 + 1];
  newDst.m[10] = v2 * m.m[2 * 4 + 2];
#endif

  return newDst;
}

// Mat3 scaling
/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has two
 * entries.
 * @param v - A vector of
 *     2 entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scaling)
(const WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  newDst.m[0] = v.v[0];
  // newDst.m[1] = 1;
  // newDst.m[2] = 0;
  // newDst.m[4] = 0;
  newDst.m[5] = v.v[1];
  // newDst.m[6] = 0;
  // newDst.m[8] = 0;
  // newDst.m[9] = 0;
  newDst.m[10] = 1;
  return newDst;
}

/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     3 entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scaling3D)(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

  newDst.m[0] = v.v[0];
  // newDst.m[1] = 0;
  // newDst.m[2] = 0;
  // newDst.m[4] = 0;
  newDst.m[5] = v.v[1];
  // newDst.m[6] = 0;
  // newDst.m[8] = 0;
  // newDst.m[9] = 0;
  newDst.m[10] = v.v[2];

  return newDst;
}

// Mat3 uniform_scale
/**
 * Scales the given 3-by-3 matrix in the X and Y dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @returns The scaled matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scale)
(WMATH_TYPE(Mat3) m, float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process first two rows with SIMD
  __m128 vec_s = _mm_set1_ps(s);
  __m128 row0 = _mm_loadu_ps(&m.m[0]); // Load first row [m00, m01, m02, pad]
  __m128 row1 = _mm_loadu_ps(&m.m[4]); // Load second row [m10, m11, m12, pad]

  // Scale the first two rows
  __m128 scaled_row0 = _mm_mul_ps(row0, vec_s);
  __m128 scaled_row1 = _mm_mul_ps(row1, vec_s);

  // Store results
  _mm_storeu_ps(&newDst.m[0], scaled_row0);
  _mm_storeu_ps(&newDst.m[4], scaled_row1);

  // Copy the third row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process first two rows with SIMD
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t row0 = vld1q_f32(&m.m[0]); // Load first row [m00, m01, m02, pad]
  float32x4_t row1 = vld1q_f32(&m.m[4]); // Load second row [m10, m11, m12, pad]

  // Scale the first two rows
  float32x4_t scaled_row0 = vmulq_f32(row0, vec_s);
  float32x4_t scaled_row1 = vmulq_f32(row1, vec_s);

  // Store results
  vst1q_f32(&newDst.m[0], scaled_row0);
  vst1q_f32(&newDst.m[4], scaled_row1);

  // Copy the third row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation - process first two rows with SIMD
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t row0 = __riscv_vle32_v_f32m1(&m.m[0], vl); // Load first row [m00, m01, m02, pad]
  vfloat32m1_t row1 = __riscv_vle32_v_f32m1(&m.m[4], vl); // Load second row [m10, m11, m12, pad]

  // Scale the first two rows
  vfloat32m1_t scaled_row0 = __riscv_vfmul_vv_f32m1(row0, vec_s, vl);
  vfloat32m1_t scaled_row1 = __riscv_vfmul_vv_f32m1(row1, vec_s, vl);

  // Store results
  __riscv_vse32_v_f32m1(&newDst.m[0], scaled_row0, vl);
  __riscv_vse32_v_f32m1(&newDst.m[4], scaled_row1, vl);

  // Copy the third row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - process first two rows with SIMD
  __m128 vec_s = __lsx_vreplgr2vr_s(s);
  __m128 row0 = __lsx_vld(&m.m[0], 0); // Load first row [m00, m01, m02, pad]
  __m128 row1 = __lsx_vld(&m.m[4], 0); // Load second row [m10, m11, m12, pad]

  // Scale the first two rows
  __m128 scaled_row0 = __lsx_vfmul_s(row0, vec_s);
  __m128 scaled_row1 = __lsx_vfmul_s(row1, vec_s);

  // Store results
  __lsx_vst(scaled_row0, &newDst.m[0], 0);
  __lsx_vst(scaled_row1, &newDst.m[4], 0);

  // Copy the third row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - process first two rows with SIMD
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t row0 = wasm_v128_load(&m.m[0]); // Load first row [m00, m01, m02, pad]
  v128_t row1 = wasm_v128_load(&m.m[4]); // Load second row [m10, m11, m12, pad]

  // Scale the first two rows
  v128_t scaled_row0 = wasm_f32x4_mul(row0, vec_s);
  v128_t scaled_row1 = wasm_f32x4_mul(row1, vec_s);

  // Store results
  wasm_v128_store(&newDst.m[0], scaled_row0);
  wasm_v128_store(&newDst.m[4], scaled_row1);

  // Copy the third row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }

#else
  // Scalar fallback with optimized loop unrolling
  newDst.m[0] = s * m.m[0];
  newDst.m[1] = s * m.m[1];
  newDst.m[2] = s * m.m[2];
  newDst.m[4] = s * m.m[4];
  newDst.m[5] = s * m.m[5];
  newDst.m[6] = s * m.m[6];
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }
#endif

  return newDst;
}

// Mat3 uniform_scale3D
/**
 * Scales the given 3-by-3 matrix in each dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @returns The scaled matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scale_3D)
(WMATH_TYPE(Mat3) m, float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all three rows with SIMD
  __m128 vec_s = _mm_set1_ps(s);
  __m128 row0 = _mm_loadu_ps(&m.m[0]); // Load first row [m00, m01, m02, pad]
  __m128 row1 = _mm_loadu_ps(&m.m[4]); // Load second row [m10, m11, m12, pad]
  __m128 row2 = _mm_loadu_ps(&m.m[8]); // Load third row [m20, m21, m22, pad]

  // Scale all three rows
  __m128 scaled_row0 = _mm_mul_ps(row0, vec_s);
  __m128 scaled_row1 = _mm_mul_ps(row1, vec_s);
  __m128 scaled_row2 = _mm_mul_ps(row2, vec_s);

  // Store results
  _mm_storeu_ps(&newDst.m[0], scaled_row0);
  _mm_storeu_ps(&newDst.m[4], scaled_row1);
  _mm_storeu_ps(&newDst.m[8], scaled_row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all three rows with SIMD
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t row0 = vld1q_f32(&m.m[0]); // Load first row [m00, m01, m02, pad]
  float32x4_t row1 = vld1q_f32(&m.m[4]); // Load second row [m10, m11, m12, pad]
  float32x4_t row2 = vld1q_f32(&m.m[8]); // Load third row [m20, m21, m22, pad]

  // Scale all three rows
  float32x4_t scaled_row0 = vmulq_f32(row0, vec_s);
  float32x4_t scaled_row1 = vmulq_f32(row1, vec_s);
  float32x4_t scaled_row2 = vmulq_f32(row2, vec_s);

  // Store results
  vst1q_f32(&newDst.m[0], scaled_row0);
  vst1q_f32(&newDst.m[4], scaled_row1);
  vst1q_f32(&newDst.m[8], scaled_row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation - process all three rows with SIMD
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t row0 = __riscv_vle32_v_f32m1(&m.m[0], vl); // Load first row [m00, m01, m02, pad]
  vfloat32m1_t row1 = __riscv_vle32_v_f32m1(&m.m[4], vl); // Load second row [m10, m11, m12, pad]
  vfloat32m1_t row2 = __riscv_vle32_v_f32m1(&m.m[8], vl); // Load third row [m20, m21, m22, pad]

  // Scale all three rows
  vfloat32m1_t scaled_row0 = __riscv_vfmul_vv_f32m1(row0, vec_s, vl);
  vfloat32m1_t scaled_row1 = __riscv_vfmul_vv_f32m1(row1, vec_s, vl);
  vfloat32m1_t scaled_row2 = __riscv_vfmul_vv_f32m1(row2, vec_s, vl);

  // Store results
  __riscv_vse32_v_f32m1(&newDst.m[0], scaled_row0, vl);
  __riscv_vse32_v_f32m1(&newDst.m[4], scaled_row1, vl);
  __riscv_vse32_v_f32m1(&newDst.m[8], scaled_row2, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - process all three rows with SIMD
  __m128 vec_s = __lsx_vreplgr2vr_s(s);
  __m128 row0 = __lsx_vld(&m.m[0], 0); // Load first row [m00, m01, m02, pad]
  __m128 row1 = __lsx_vld(&m.m[4], 0); // Load second row [m10, m11, m12, pad]
  __m128 row2 = __lsx_vld(&m.m[8], 0); // Load third row [m20, m21, m22, pad]

  // Scale all three rows
  __m128 scaled_row0 = __lsx_vfmul_s(row0, vec_s);
  __m128 scaled_row1 = __lsx_vfmul_s(row1, vec_s);
  __m128 scaled_row2 = __lsx_vfmul_s(row2, vec_s);

  // Store results
  __lsx_vst(scaled_row0, &newDst.m[0], 0);
  __lsx_vst(scaled_row1, &newDst.m[4], 0);
  __lsx_vst(scaled_row2, &newDst.m[8], 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - process all three rows with SIMD
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t row0 = wasm_v128_load(&m.m[0]); // Load first row [m00, m01, m02, pad]
  v128_t row1 = wasm_v128_load(&m.m[4]); // Load second row [m10, m11, m12, pad]
  v128_t row2 = wasm_v128_load(&m.m[8]); // Load third row [m20, m21, m22, pad]

  // Scale all three rows
  v128_t scaled_row0 = wasm_f32x4_mul(row0, vec_s);
  v128_t scaled_row1 = wasm_f32x4_mul(row1, vec_s);
  v128_t scaled_row2 = wasm_f32x4_mul(row2, vec_s);

  // Store results
  wasm_v128_store(&newDst.m[0], scaled_row0);
  wasm_v128_store(&newDst.m[4], scaled_row1);
  wasm_v128_store(&newDst.m[8], scaled_row2);

#else
  // Scalar fallback with optimized loop unrolling
  newDst.m[0] = s * m.m[0];
  newDst.m[1] = s * m.m[1];
  newDst.m[2] = s * m.m[2];
  newDst.m[4] = s * m.m[4];
  newDst.m[5] = s * m.m[5];
  newDst.m[6] = s * m.m[6];
  newDst.m[8] = s * m.m[8];
  newDst.m[9] = s * m.m[9];
  newDst.m[10] = s * m.m[10];
#endif

  return newDst;
}

// Mat3 uniform_scaling
/**
 * Creates a 3-by-3 matrix which scales uniformly in the X and Y dimensions
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling)
(float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - create uniform scaling matrix efficiently
  __m128 vec_s = _mm_set1_ps(s);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);

  // Create rows: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, 1, pad]
  __m128 row0 =
      _mm_move_ss(vec_s, vec_zero); // [0, s, 0, 0] -> shuffle to [s, 0, 0, pad]
  row0 =
      _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0)); // [s, 0, 0, pad]

  __m128 row1 =
      _mm_move_ss(vec_zero, vec_s); // [s, 0, 0, 0] -> shuffle to [0, s, 0, pad]
  row1 =
      _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1)); // [0, s, 0, pad]

  __m128 row2 = _mm_move_ss(
      vec_zero, vec_one); // [1, 0, 0, 0] -> shuffle to [0, 0, 1, pad]
  row2 =
      _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0)); // [0, 0, 1, pad]

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - create uniform scaling matrix efficiently
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Create rows and store results directly
  float32x4_t row0 = vec_s;
  row0 = vsetq_lane_f32(0.0f, row0, 1);
  row0 = vsetq_lane_f32(0.0f, row0, 2);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(s, row1, 1);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(1.0f, row2, 2);

  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation - create uniform scaling matrix efficiently
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vec_one = __riscv_vfmv_v_f_f32m1(1.0f, vl);

  // Create rows: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, 1, pad]
  // Manual setting because we need precise element control
  __riscv_vse32_v_f32m1(&newDst.m[0], vec_s, vl);
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;

  __riscv_vse32_v_f32m1(&newDst.m[4], vec_zero, vl);
  newDst.m[5] = s;
  newDst.m[6] = 0.0f;

  __riscv_vse32_v_f32m1(&newDst.m[8], vec_zero, vl);
  newDst.m[9] = 0.0f;
  newDst.m[10] = 1.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - create uniform scaling matrix efficiently
  __m128 vec_s = __lsx_vreplgr2vr_s(s);
  __m128 vec_zero = __lsx_vldi(0); // Load immediate zero
  __m128 vec_one = __lsx_vreplgr2vr_s(1.0f);

  // Create rows: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, 1, pad]
  __m128 row0 = __lsx_vilvl_w(vec_zero, vec_s);  // [s, 0, pad, pad]
  row0 = __lsx_vori_b(row0, row0, 0);  // Manually set for [s, 0, 0, pad] - since we need more control
  // Manual setting using individual operations
  __lsx_vstelm_w(vec_s, &newDst.m[0], 0, 0); // Set m[0] = s
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;

  __lsx_vstelm_w(vec_zero, &newDst.m[4], 0, 0);  // Set m[4] = 0
  __lsx_vstelm_w(vec_s, &newDst.m[4], 4, 1);    // Set m[5] = s
  __lsx_vstelm_w(vec_zero, &newDst.m[4], 8, 2); // Set m[6] = 0

  __lsx_vstelm_w(vec_zero, &newDst.m[8], 0, 0);  // Set m[8] = 0
  __lsx_vstelm_w(vec_zero, &newDst.m[8], 4, 1);  // Set m[9] = 0
  __lsx_vstelm_w(vec_one, &newDst.m[8], 8, 2);   // Set m[10] = 1

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - create uniform scaling matrix efficiently
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t vec_zero = wasm_f32x4_splat(0.0f);
  v128_t vec_one = wasm_f32x4_splat(1.0f);

  // Create rows: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, 1, pad]
  wasm_v128_store(&newDst.m[0], vec_s);
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;

  wasm_v128_store(&newDst.m[4], vec_zero);
  newDst.m[5] = s;

  wasm_v128_store(&newDst.m[8], vec_zero);
  newDst.m[10] = 1.0f;

#else
  // Scalar fallback - zero initialization is more efficient than memset
  newDst.m[0] = s;
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;
  newDst.m[4] = 0.0f;
  newDst.m[5] = s;
  newDst.m[6] = 0.0f;
  newDst.m[8] = 0.0f;
  newDst.m[9] = 0.0f;
  newDst.m[10] = 1.0f;
#endif

  return newDst;
}

// Mat3 uniform_scaling3D
/**
 * Creates a 3-by-3 matrix which scales uniformly in each dimension
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling_3D)
(float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - create uniform 3D scaling matrix efficiently
  __m128 vec_s = _mm_set1_ps(s);
  __m128 vec_zero = _mm_setzero_ps();

  // Create diagonal matrix with s on diagonal
  // Row0: [s, 0, 0, pad]
  __m128 row0 = _mm_move_ss(vec_s, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, s, 0, pad]
  __m128 row1 = _mm_move_ss(vec_zero, vec_s);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Row2: [0, 0, s, pad]
  __m128 row2 = _mm_move_ss(vec_zero, vec_s);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - create uniform 3D scaling matrix efficiently
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);

  // Create diagonal matrix with s on diagonal
  float32x4_t row0 = vec_s;
  row0 = vsetq_lane_f32(0.0f, row0, 1);
  row0 = vsetq_lane_f32(0.0f, row0, 2);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(s, row1, 1);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(s, row2, 2);

  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation - create uniform 3D scaling matrix efficiently
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  // Create diagonal matrix with s on diagonal: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, s, pad]
  __riscv_vse32_v_f32m1(&newDst.m[0], vec_s, vl);
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;

  __riscv_vse32_v_f32m1(&newDst.m[4], vec_zero, vl);
  newDst.m[5] = s;
  newDst.m[6] = 0.0f;

  __riscv_vse32_v_f32m1(&newDst.m[8], vec_zero, vl);
  newDst.m[9] = 0.0f;
  newDst.m[10] = s;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - create uniform 3D scaling matrix efficiently
  __m128 vec_s = __lsx_vreplgr2vr_s(s);
  __m128 vec_zero = __lsx_vldi(0); // Load immediate zero

  // Create diagonal matrix: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, s, pad]
  __lsx_vstelm_w(vec_s, &newDst.m[0], 0, 0); // Set m[0] = s
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;

  __lsx_vstelm_w(vec_zero, &newDst.m[4], 0, 0);  // Set m[4] = 0
  __lsx_vstelm_w(vec_s, &newDst.m[4], 4, 1);    // Set m[5] = s
  newDst.m[6] = 0.0f;

  __lsx_vstelm_w(vec_zero, &newDst.m[8], 0, 0);  // Set m[8] = 0
  __lsx_vstelm_w(vec_zero, &newDst.m[8], 4, 1);  // Set m[9] = 0
  __lsx_vstelm_w(vec_s, &newDst.m[8], 8, 2);     // Set m[10] = s

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - create uniform 3D scaling matrix efficiently
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t vec_zero = wasm_f32x4_splat(0.0f);

  // Create diagonal matrix: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, s, pad]
  wasm_v128_store(&newDst.m[0], vec_s);
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;

  wasm_v128_store(&newDst.m[4], vec_zero);
  newDst.m[5] = s;
  newDst.m[6] = 0.0f;

  wasm_v128_store(&newDst.m[8], vec_zero);
  newDst.m[9] = 0.0f;
  newDst.m[10] = s;

#else
  // Scalar fallback - direct assignment is more efficient
  newDst.m[0] = s;
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;
  newDst.m[4] = 0.0f;
  newDst.m[5] = s;
  newDst.m[6] = 0.0f;
  newDst.m[8] = 0.0f;
  newDst.m[9] = 0.0f;
  newDst.m[10] = s;
#endif

  return newDst;
}

// Mat4 getScaling
/**
 * Returns the "3d" scaling component of the matrix
 * @param m - The Matrix
 */
WMATH_TYPE(Vec3)
WMATH_CALL(Mat4, get_scaling)(WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation: directly read columns like scalar version

  // x column: m[0], m[1], m[2]
  __m128 col0 = _mm_set_ps(0.0f, m.m[2], m.m[1], m.m[0]);
  __m128 sq0 = _mm_mul_ps(col0, col0);
  result.v[0] = sqrtf(_mm_cvtss_f32(wcn_hadd_ps(sq0)));

  // y column: m[4], m[5], m[6]
  __m128 col1 = _mm_set_ps(0.0f, m.m[6], m.m[5], m.m[4]);
  __m128 sq1 = _mm_mul_ps(col1, col1);
  result.v[1] = sqrtf(_mm_cvtss_f32(wcn_hadd_ps(sq1)));

  // z column: m[8], m[9], m[10]
  __m128 col2 = _mm_set_ps(0.0f, m.m[10], m.m[9], m.m[8]);
  __m128 sq2 = _mm_mul_ps(col2, col2);
  result.v[2] = sqrtf(_mm_cvtss_f32(wcn_hadd_ps(sq2)));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation

  float32x4_t col0 = { m.m[0], m.m[1], m.m[2], 0.0f };
  float32x4_t sq0  = vmulq_f32(col0, col0);
  result.v[0] = sqrtf(wcn_hadd_f32(sq0));

  float32x4_t col1 = { m.m[4], m.m[5], m.m[6], 0.0f };
  float32x4_t sq1  = vmulq_f32(col1, col1);
  result.v[1] = sqrtf(wcn_hadd_f32(sq1));

  float32x4_t col2 = { m.m[8], m.m[9], m.m[10], 0.0f };
  float32x4_t sq2  = vmulq_f32(col2, col2);
  result.v[2] = sqrtf(wcn_hadd_f32(sq2));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);

  // x column: m.m[0], m.m[1], m.m[2]
  float x_values[4] = { m.m[0], m.m[1], m.m[2], 0.0f };
  vfloat32m1_t col0 = __riscv_vle32_v_f32m1(x_values, vl);
  vfloat32m1_t sq0 = __riscv_vfmul_vv_f32m1(col0, col0, vl);
  // Sum first 3 elements of squared vector
  float32_t x_sum = __riscv_vfmv_s_f_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(sq0, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl), vl);
  result.v[0] = sqrtf(x_sum);

  // y column: m.m[4], m.m[5], m.m[6]
  float y_values[4] = { m.m[4], m.m[5], m.m[6], 0.0f };
  vfloat32m1_t col1 = __riscv_vle32_v_f32m1(y_values, vl);
  vfloat32m1_t sq1 = __riscv_vfmul_vv_f32m1(col1, col1, vl);
  float32_t y_sum = __riscv_vfmv_s_f_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(sq1, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl), vl);
  result.v[1] = sqrtf(y_sum);

  // z column: m.m[8], m.m[9], m.m[10]
  float z_values[4] = { m.m[8], m.m[9], m.m[10], 0.0f };
  vfloat32m1_t col2 = __riscv_vle32_v_f32m1(z_values, vl);
  vfloat32m1_t sq2 = __riscv_vfmul_vv_f32m1(col2, col2, vl);
  float32_t z_sum = __riscv_vfmv_s_f_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(sq2, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl), vl);
  result.v[2] = sqrtf(z_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  // For this operation, use the more efficient scalar approach since we need individual
  // element access for the length calculation
  float x_sq = m.m[0]*m.m[0] + m.m[1]*m.m[1] + m.m[2]*m.m[2];
  float y_sq = m.m[4]*m.m[4] + m.m[5]*m.m[5] + m.m[6]*m.m[6];
  float z_sq = m.m[8]*m.m[8] + m.m[9]*m.m[9] + m.m[10]*m.m[10];
  result.v[0] = sqrtf(x_sq);
  result.v[1] = sqrtf(y_sq);
  result.v[2] = sqrtf(z_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t col0 = wasm_v128_load(&m.m[0]);  // Load 4 floats starting from m[0]
  // Create [m.m[0], m.m[1], m.m[2], 0.0f] using splat and replace
  v128_t col0_vals = wasm_v128_const_f32(m.m[0], m.m[1], m.m[2], 0.0f);
  v128_t sq0 = wasm_f32x4_mul(col0_vals, col0_vals);
  float x_sum = sqrtf(wasm_f32x4_extract_lane(sq0, 0) + wasm_f32x4_extract_lane(sq0, 1) + wasm_f32x4_extract_lane(sq0, 2));
  result.v[0] = x_sum;

  v128_t col1_vals = wasm_v128_const_f32(m.m[4], m.m[5], m.m[6], 0.0f);
  v128_t sq1 = wasm_f32x4_mul(col1_vals, col1_vals);
  float y_sum = sqrtf(wasm_f32x4_extract_lane(sq1, 0) + wasm_f32x4_extract_lane(sq1, 1) + wasm_f32x4_extract_lane(sq1, 2));
  result.v[1] = y_sum;

  v128_t col2_vals = wasm_v128_const_f32(m.m[8], m.m[9], m.m[10], 0.0f);
  v128_t sq2 = wasm_f32x4_mul(col2_vals, col2_vals);
  float z_sum = sqrtf(wasm_f32x4_extract_lane(sq2, 0) + wasm_f32x4_extract_lane(sq2, 1) + wasm_f32x4_extract_lane(sq2, 2));
  result.v[2] = z_sum;

#else
  // Scalar fallback
  result.v[0] = sqrtf(m.m[0]*m.m[0] + m.m[1]*m.m[1] + m.m[2]*m.m[2]);
  result.v[1] = sqrtf(m.m[4]*m.m[4] + m.m[5]*m.m[5] + m.m[6]*m.m[6]);
  result.v[2] = sqrtf(m.m[8]*m.m[8] + m.m[9]*m.m[9] + m.m[10]*m.m[10]);
#endif

  return result;
}

// Mat4 scaling
/**
 * Creates a 4-by-4 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     three entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, scaling)(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - column-major scaling matrix
  __m128 col0 = _mm_set_ps(0.0f, 0.0f, 0.0f, v.v[0]); // first column
  __m128 col1 = _mm_set_ps(0.0f, 0.0f, v.v[1], 0.0f); // second column
  __m128 col2 = _mm_set_ps(0.0f, v.v[2], 0.0f, 0.0f); // third column
  __m128 col3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);   // fourth column

  _mm_storeu_ps(&newDst.m[0], col0);
  _mm_storeu_ps(&newDst.m[4], col1);
  _mm_storeu_ps(&newDst.m[8], col2);
  _mm_storeu_ps(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - column-major scaling matrix
  float32x4_t col0 = {v.v[0], 0.0f, 0.0f, 0.0f};
  float32x4_t col1 = {0.0f, v.v[1], 0.0f, 0.0f};
  float32x4_t col2 = {0.0f, 0.0f, v.v[2], 0.0f};
  float32x4_t col3 = {0.0f, 0.0f, 0.0f, 1.0f};

  vst1q_f32(&newDst.m[0], col0);
  vst1q_f32(&newDst.m[4], col1);
  vst1q_f32(&newDst.m[8], col2);
  vst1q_f32(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation - column-major scaling matrix
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t col0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);  // [0, 0, 0, 0]
  col0 = __riscv_vfslide1up_vf_f32m1(col0, v.v[0], vl);  // [v.v[0], 0, 0, 0]
  __riscv_vse32_v_f32m1(&newDst.m[0], col0, vl);

  vfloat32m1_t col1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);  // [0, 0, 0, 0]
  col1 = __riscv_vfslide1up_vf_f32m1(col1, v.v[1], vl);  // But we need [0, v.v[1], 0, 0], so use manual set
  newDst.m[0] = 0.0f;
  newDst.m[1] = v.v[1];
  newDst.m[2] = 0.0f;
  newDst.m[3] = 0.0f;

  newDst.m[4] = 0.0f;
  newDst.m[5] = 0.0f;
  newDst.m[6] = v.v[2];
  newDst.m[7] = 0.0f;

  newDst.m[8] = 0.0f;
  newDst.m[9] = 0.0f;
  newDst.m[10] = 0.0f;
  newDst.m[11] = 0.0f;

  newDst.m[12] = 0.0f;
  newDst.m[13] = 0.0f;
  newDst.m[14] = 0.0f;
  newDst.m[15] = 1.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - column-major scaling matrix
  // Using scalar approach since we need to set specific elements
  newDst.m[0]  = v.v[0];
  newDst.m[1]  = 0.0f;
  newDst.m[2]  = 0.0f;
  newDst.m[3]  = 0.0f;

  newDst.m[4]  = 0.0f;
  newDst.m[5]  = v.v[1];
  newDst.m[6]  = 0.0f;
  newDst.m[7]  = 0.0f;

  newDst.m[8]  = 0.0f;
  newDst.m[9]  = 0.0f;
  newDst.m[10] = v.v[2];
  newDst.m[11] = 0.0f;

  newDst.m[12] = 0.0f;
  newDst.m[13] = 0.0f;
  newDst.m[14] = 0.0f;
  newDst.m[15] = 1.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - column-major scaling matrix
  v128_t col0 = wasm_v128_const_f32(v.v[0], 0.0f, 0.0f, 0.0f);
  v128_t col1 = wasm_v128_const_f32(0.0f, v.v[1], 0.0f, 0.0f);
  v128_t col2 = wasm_v128_const_f32(0.0f, 0.0f, v.v[2], 0.0f);
  v128_t col3 = wasm_v128_const_f32(0.0f, 0.0f, 0.0f, 1.0f);

  wasm_v128_store(&newDst.m[0], col0);
  wasm_v128_store(&newDst.m[4], col1);
  wasm_v128_store(&newDst.m[8], col2);
  wasm_v128_store(&newDst.m[12], col3);

#else
  // Scalar fallback
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0]  = v.v[0];
  newDst.m[5]  = v.v[1];
  newDst.m[10] = v.v[2];
  newDst.m[15] = 1.0f;
#endif

  return newDst;
}

// Mat4 uniformScale
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, uniform_scale)
(WMATH_TYPE(Mat4) m, float s) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process first three rows with SIMD
  __m128 vec_s = _mm_set1_ps(s);
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Scale the first three rows (rotation/scaling part)
  __m128 scaled_row0 = _mm_mul_ps(row0, vec_s);
  __m128 scaled_row1 = _mm_mul_ps(row1, vec_s);
  __m128 scaled_row2 = _mm_mul_ps(row2, vec_s);

  // Store results
  wcn_mat4_set_row(&newDst, 0, scaled_row0);
  wcn_mat4_set_row(&newDst, 1, scaled_row1);
  wcn_mat4_set_row(&newDst, 2, scaled_row2);

  // Copy the fourth row (translation part) unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process first three rows with SIMD
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Scale the first three rows (rotation/scaling part)
  float32x4_t scaled_row0 = vmulq_f32(row0, vec_s);
  float32x4_t scaled_row1 = vmulq_f32(row1, vec_s);
  float32x4_t scaled_row2 = vmulq_f32(row2, vec_s);

  // Store results
  wcn_mat4_set_row(&newDst, 0, scaled_row0);
  wcn_mat4_set_row(&newDst, 1, scaled_row1);
  wcn_mat4_set_row(&newDst, 2, scaled_row2);

  // Copy the fourth row (translation part) unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation - process first three rows with SIMD
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t row0 = wcn_mat4_get_row(&m, 0);
  vfloat32m1_t row1 = wcn_mat4_get_row(&m, 1);
  vfloat32m1_t row2 = wcn_mat4_get_row(&m, 2);
  vfloat32m1_t row3 = wcn_mat4_get_row(&m, 3);

  // Scale the first three rows (rotation/scaling part)
  vfloat32m1_t scaled_row0 = __riscv_vfmul_vv_f32m1(row0, vec_s, vl);
  vfloat32m1_t scaled_row1 = __riscv_vfmul_vv_f32m1(row1, vec_s, vl);
  vfloat32m1_t scaled_row2 = __riscv_vfmul_vv_f32m1(row2, vec_s, vl);

  // Store results
  wcn_mat4_set_row(&newDst, 0, scaled_row0);
  wcn_mat4_set_row(&newDst, 1, scaled_row1);
  wcn_mat4_set_row(&newDst, 2, scaled_row2);

  // Copy the fourth row (translation part) unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - process first three rows with SIMD
  __m128 vec_s = __lsx_vreplgr2vr_s(s);
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Scale the first three rows (rotation/scaling part)
  __m128 scaled_row0 = __lsx_vfmul_s(row0, vec_s);
  __m128 scaled_row1 = __lsx_vfmul_s(row1, vec_s);
  __m128 scaled_row2 = __lsx_vfmul_s(row2, vec_s);

  // Store results
  wcn_mat4_set_row(&newDst, 0, scaled_row0);
  wcn_mat4_set_row(&newDst, 1, scaled_row1);
  wcn_mat4_set_row(&newDst, 2, scaled_row2);

  // Copy the fourth row (translation part) unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - process first three rows with SIMD
  v128_t vec_s = wasm_f32x4_splat(s);
  v128_t row0 = wcn_mat4_get_row(&m, 0);
  v128_t row1 = wcn_mat4_get_row(&m, 1);
  v128_t row2 = wcn_mat4_get_row(&m, 2);
  v128_t row3 = wcn_mat4_get_row(&m, 3);

  // Scale the first three rows (rotation/scaling part)
  v128_t scaled_row0 = wasm_f32x4_mul(row0, vec_s);
  v128_t scaled_row1 = wasm_f32x4_mul(row1, vec_s);
  v128_t scaled_row2 = wasm_f32x4_mul(row2, vec_s);

  // Store results
  wcn_mat4_set_row(&newDst, 0, scaled_row0);
  wcn_mat4_set_row(&newDst, 1, scaled_row1);
  wcn_mat4_set_row(&newDst, 2, scaled_row2);

  // Copy the fourth row (translation part) unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized loop unrolling
  newDst.m[0] = s * m.m[0 * 4 + 0];
  newDst.m[1] = s * m.m[0 * 4 + 1];
  newDst.m[2] = s * m.m[0 * 4 + 2];
  newDst.m[3] = s * m.m[0 * 4 + 3];
  newDst.m[4] = s * m.m[1 * 4 + 0];
  newDst.m[5] = s * m.m[1 * 4 + 1];
  newDst.m[6] = s * m.m[1 * 4 + 2];
  newDst.m[7] = s * m.m[1 * 4 + 3];
  newDst.m[8] = s * m.m[2 * 4 + 0];
  newDst.m[9] = s * m.m[2 * 4 + 1];
  newDst.m[10] = s * m.m[2 * 4 + 2];
  newDst.m[11] = s * m.m[2 * 4 + 3];

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 uniformScaling
/**
 * Creates a 4-by-4 matrix which scales a uniform amount in each dimension.
 * @param s - the amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, uniform_scaling)
(float s) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - create uniform scaling matrix efficiently
  __m128 vec_s = _mm_set1_ps(s);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);

  // Create diagonal matrix with s on diagonal
  // Row0: [s, 0, 0, 0]
  __m128 row0 = _mm_move_ss(vec_s, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, s, 0, 0]
  __m128 row1 = _mm_move_ss(vec_zero, vec_s);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Row2: [0, 0, s, 0]
  __m128 row2 = _mm_move_ss(vec_zero, vec_s);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Row3: [0, 0, 0, 1]
  __m128 row3 = _mm_move_ss(vec_zero, vec_one);
  row3 = _mm_shuffle_ps(row3, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);
  _mm_storeu_ps(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - create uniform scaling matrix efficiently
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Create diagonal matrix with s on diagonal
  float32x4_t row0 = vec_zero;
  row0 = vsetq_lane_f32(s, row0, 0);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(s, row1, 1);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(s, row2, 2);

  float32x4_t row3 = vec_zero;
  row3 = vsetq_lane_f32(1.0f, row3, 3);

  // Store results
  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);
  vst1q_f32(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV
  // RISC-V Vector Extension implementation - create uniform scaling matrix efficiently
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t vec_one = __riscv_vfmv_v_f_f32m1(1.0f, vl);

  // Create diagonal matrix: [s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]
  __riscv_vse32_v_f32m1(&newDst.m[0], vec_s, vl);  // Row 0: [s, 0, 0, 0] - just set first element
  newDst.m[0] = s; newDst.m[1] = 0.0f; newDst.m[2] = 0.0f; newDst.m[3] = 0.0f;

  newDst.m[4] = 0.0f; newDst.m[5] = s; newDst.m[6] = 0.0f; newDst.m[7] = 0.0f;

  newDst.m[8] = 0.0f; newDst.m[9] = 0.0f; newDst.m[10] = s; newDst.m[11] = 0.0f;

  newDst.m[12] = 0.0f; newDst.m[13] = 0.0f; newDst.m[14] = 0.0f; newDst.m[15] = 1.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - create uniform scaling matrix efficiently
  __m128 vec_s = __lsx_vreplgr2vr_s(s);
  __m128 vec_zero = __lsx_vldi(0); // Load immediate zero
  __m128 vec_one = __lsx_vreplgr2vr_s(1.0f);

  // Create diagonal matrix using direct element assignment
  newDst.m[0] = s; newDst.m[1] = 0.0f; newDst.m[2] = 0.0f; newDst.m[3] = 0.0f;
  newDst.m[4] = 0.0f; newDst.m[5] = s; newDst.m[6] = 0.0f; newDst.m[7] = 0.0f;
  newDst.m[8] = 0.0f; newDst.m[9] = 0.0f; newDst.m[10] = s; newDst.m[11] = 0.0f;
  newDst.m[12] = 0.0f; newDst.m[13] = 0.0f; newDst.m[14] = 0.0f; newDst.m[15] = 1.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - create uniform scaling matrix efficiently
  v128_t col0 = wasm_v128_const_f32(s, 0.0f, 0.0f, 0.0f);
  v128_t col1 = wasm_v128_const_f32(0.0f, s, 0.0f, 0.0f);
  v128_t col2 = wasm_v128_const_f32(0.0f, 0.0f, s, 0.0f);
  v128_t col3 = wasm_v128_const_f32(0.0f, 0.0f, 0.0f, 1.0f);

  // Store results
  wasm_v128_store(&newDst.m[0], col0);
  wasm_v128_store(&newDst.m[4], col1);
  wasm_v128_store(&newDst.m[8], col2);
  wasm_v128_store(&newDst.m[12], col3);

#else
  // Scalar fallback - direct assignment is more efficient than memset
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0] = s;
  newDst.m[5] = s;
  newDst.m[10] = s;
  newDst.m[15] = 1.0f;
#endif

  return newDst;
}

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, ortho)
(const float left, const float right, const float bottom, const float top, const float near, const float far) {
  WMATH_TYPE(Mat4) newDst;

  // 1. 预先计算标量 (Scalar Pre-calculation)
  // 这种数学运算由标量单元处理非常快，且允许编译器进行指令调度优化。

  const float r_l = right - left;
  const float t_b = top - bottom;
  const float f_n = far - near;

  float tx = -(right + left) / r_l;
  float ty = -(top + bottom) / t_b;
  float tz = -(far + near) / f_n;

  float m00 = 2.0f / r_l;
  float m11 = 2.0f / t_b;
  float m22 = -2.0f / f_n;

  // 矩阵内存布局 (Column-Major):
  // Col 0: [m00, 0, 0, 0]
  // Col 1: [0, m11, 0, 0]
  // Col 2: [0, 0, m22, 0]
  // Col 3: [tx, ty, tz, 1]

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE Implementation
  // _mm_setr_ps 按参数顺序构建向量 [e0, e1, e2, e3]

  __m128 col0 = _mm_setr_ps(m00, 0.0f, 0.0f, 0.0f);
  __m128 col1 = _mm_setr_ps(0.0f, m11, 0.0f, 0.0f);
  __m128 col2 = _mm_setr_ps(0.0f, 0.0f, m22, 0.0f);
  __m128 col3 = _mm_setr_ps(tx, ty, tz, 1.0f);

  _mm_storeu_ps(&newDst.m[0], col0);
  _mm_storeu_ps(&newDst.m[4], col1);
  _mm_storeu_ps(&newDst.m[8], col2);
  _mm_storeu_ps(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Implementation

  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, m22, 0.0f};
  float c3[4] = {tx, ty, tz, 1.0f};

  vst1q_f32(&newDst.m[0], vld1q_f32(c0));
  vst1q_f32(&newDst.m[4], vld1q_f32(c1));
  vst1q_f32(&newDst.m[8], vld1q_f32(c2));
  vst1q_f32(&newDst.m[12], vld1q_f32(c3));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD Implementation

  v128_t col0 = wasm_f32x4_make(m00, 0.0f, 0.0f, 0.0f);
  v128_t col1 = wasm_f32x4_make(0.0f, m11, 0.0f, 0.0f);
  v128_t col2 = wasm_f32x4_make(0.0f, 0.0f, m22, 0.0f);
  v128_t col3 = wasm_f32x4_make(tx, ty, tz, 1.0f);

  wasm_v128_store(&newDst.m[0], col0);
  wasm_v128_store(&newDst.m[4], col1);
  wasm_v128_store(&newDst.m[8], col2);
  wasm_v128_store(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Implementation

  float data[16] = {
      m00, 0.0f, 0.0f, 0.0f,
      0.0f, m11, 0.0f, 0.0f,
      0.0f, 0.0f, m22, 0.0f,
      tx,  ty,  tz,  1.0f
  };

  size_t vl = __riscv_vsetvli(16, __RISCV_VTYPE_F32, __RISCV_VLMUL_1);
  vfloat32m1_t vec_all = __riscv_vle32_v_f32m1(data, vl);
  __riscv_vse32_v_f32m1(newDst.m, vec_all, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Implementation

  float c0[4] = {m00, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, m11, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, m22, 0.0f};
  float c3[4] = {tx, ty, tz, 1.0f};

  __lsx_vst(__lsx_vld(c0, 0), &newDst.m[0], 0);
  __lsx_vst(__lsx_vld(c1, 0), &newDst.m[4], 0);
  __lsx_vst(__lsx_vld(c2, 0), &newDst.m[8], 0);
  __lsx_vst(__lsx_vld(c3, 0), &newDst.m[12], 0);

#else
  // Scalar Fallback
  // 使用 memset 确保所有未显式赋值的元素为 0
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));

  newDst.m[0] = m00;
  newDst.m[5] = m11;
  newDst.m[10] = m22;
  newDst.m[12] = tx;
  newDst.m[13] = ty;
  newDst.m[14] = tz;
  newDst.m[15] = 1.0f;
#endif

  return newDst;
}