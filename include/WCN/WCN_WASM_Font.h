#ifndef WCN_WASM_FONT_H
#define WCN_WASM_FONT_H

#include "WCN.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// WCN WASM Font Decoder
// ============================================================================

/**
 * @brief Get the WASM font decoder instance
 * 
 * This font decoder is designed for WebAssembly builds and uses JavaScript
 * interop to handle font loading and glyph rendering.
 * 
 * @return Pointer to the WASM font decoder
 */
WCN_FontDecoder* wcn_get_wasm_font_decoder(void);

/**
 * @brief Create a default font face for WASM builds
 * 
 * This function creates a default font face that can be used for text rendering.
 * 
 * @return Pointer to the default font face
 */
WCN_FontFace* wcn_wasm_create_default_font_face(void);

#ifdef __EMSCRIPTEN__
/**
 * @brief Get the WASM font decoder instance (WASM export)
 * 
 * This function is exported to JavaScript for WASM builds.
 * 
 * @return Pointer to the WASM font decoder
 */
WCN_WASM_EXPORT WCN_FontDecoder* wcn_wasm_get_font_decoder(void);

/**
 * @brief Create a default font face for WASM builds (WASM export)
 * 
 * This function is exported to JavaScript for WASM builds.
 * 
 * @return Pointer to the default font face
 */
WCN_WASM_EXPORT WCN_FontFace* wcn_wasm_create_default_font_face(void);
#endif

#ifdef __cplusplus
}
#endif

#endif // WCN_WASM_FONT_H