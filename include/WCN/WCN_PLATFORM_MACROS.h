#ifndef WCN_PLATFORM_MACROS_H
#define WCN_PLATFORM_MACROS_H

// ========================================================================
// Unified SIMD architecture detection macros
// These macros provide consistent detection across all source files
// ========================================================================

// RISC-V Vector Extension detection
#if defined(__riscv) && (defined(__riscv_vector) || defined(__riscv_vector__)) && __riscv_v_intrinsic >= 1000000
  #define WCN_HAS_RISCV_VECTOR 1
#else
  #define WCN_HAS_RISCV_VECTOR 0
#endif

// LoongArch LSX (Loongson SIMD Extension) detection
#if defined(__loongarch_sx__) || defined(__loongarch_sx)
  #define WCN_HAS_LOONGARCH_LSX 1
#else
  #define WCN_HAS_LOONGARCH_LSX 0
#endif

// LoongArch LASX (Loongson Advanced SIMD Extension) detection
#if defined(__loongarch_asx__) || defined(__loongarch_asx)
  #define WCN_HAS_LOONGARCH_LASX 1
#else
  #define WCN_HAS_LOONGARCH_LASX 0
#endif

// x86_64 SSE/AVX detection
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
  #define WCN_HAS_X86_64 1
  // AVX/AVX2 support detection
  #if defined(__AVX2__)
    #define WCN_HAS_AVX2 1
  #elif defined(__AVX__)
    #define WCN_HAS_AVX 1
  #else
    #define WCN_HAS_AVX2 0
    #define WCN_HAS_AVX 0
  #endif
#else
  #define WCN_HAS_X86_64 0
  #define WCN_HAS_AVX2 0
  #define WCN_HAS_AVX 0
#endif
// FMA (Fused Multiply-Add) support
#if defined(__FMA__)
  #define WCN_HAS_FMA 1
#else 
  #define WCN_HAS_FMA 0
#endif


// AArch64 NEON detection
#ifdef __aarch64__
  #define WCN_HAS_AARCH64 1
#else
  #define WCN_HAS_AARCH64 0
#endif

// WebAssembly SIMD detection
#ifdef __wasm_simd128__
  #define WCN_HAS_WASM_SIMD 1
#else
  #define WCN_HAS_WASM_SIMD 0
#endif

// LoongArch architecture detection (optional, for completeness)
#ifdef __loongarch__
  #define WCN_HAS_LOONGARCH 1
#else
  #define WCN_HAS_LOONGARCH 0
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define WCN_WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define WCN_WASM_EXPORT
#endif

#endif // WCN_PLATFORM_MACROS_H