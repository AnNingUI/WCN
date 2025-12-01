#ifndef WCN_WASM_IMAGE_H
#define WCN_WASM_IMAGE_H

#include "WCN/WCN.h"
#include "WCN/WCN_WASM.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the built-in WASM image decoder.
 *
 * This decoder uses the lightweight stb_image implementation compiled into the
 * WebAssembly module so that PNG/JPEG and other common bitmap formats can be
 * decoded without any JavaScript glue.
 *
 * @return Pointer to the decoder instance.
 */
WCN_ImageDecoder* wcn_get_wasm_image_decoder(void);

#ifdef __EMSCRIPTEN__
/**
 * @brief WASM export so JavaScript can retrieve the decoder pointer.
 */
WCN_WASM_EXPORT WCN_ImageDecoder* wcn_wasm_get_image_decoder(void);
#endif

#ifdef __cplusplus
}
#endif

#endif // WCN_WASM_IMAGE_H
