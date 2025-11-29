// ============================================================================
// WCN WASM Font Decoder Usage Example
// This example shows how to use the WASM font decoder in C code
// ============================================================================

#include <WCN/WCN.h>
#include <WCN/WCN_WASM_Font.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

// Forward declarations
static void render_text_example(WCN_Context* ctx);

/**
 * @brief Example of how to use the WASM font decoder
 * 
 * This function demonstrates:
 * 1. Getting the WASM font decoder instance
 * 2. Registering the decoder with a WCN context
 * 3. Using the decoder for text rendering
 */
void example_wasm_font_decoder_usage(WCN_Context* ctx) {
    printf("=== WCN WASM Font Decoder Usage Example ===\n");
    
    // Get the WASM font decoder instance
    // Note: In a real WASM build, this would be available through the exported function
#ifdef __EMSCRIPTEN__
    WCN_FontDecoder* wasm_decoder = wcn_wasm_get_font_decoder();
#else
    // For non-WASM builds, you might want to use a different decoder
    // This is just for demonstration purposes
    WCN_FontDecoder* wasm_decoder = NULL;
#endif
    
    if (wasm_decoder) {
        printf("Got WASM font decoder: %s\n", wasm_decoder->name);
        
        // Register the font decoder with the context
        wcn_register_font_decoder(ctx, wasm_decoder);
        printf("Registered WASM font decoder with context\n");
        
        // Now you can use text rendering functions
        render_text_example(ctx);
    } else {
        printf("WASM font decoder not available in this build\n");
        
        // For demonstration, we'll show how you might handle this case
        // In a real application, you might fall back to a different decoder
        // or show an error message
        printf("Text rendering will be disabled\n");
    }
}

/**
 * @brief Render text using the registered font decoder
 * 
 * @param ctx WCN context
 */
static void render_text_example(WCN_Context* ctx) {
    if (!ctx) {
        printf("Invalid context\n");
        return;
    }
    
    // Set up some text rendering state
    wcn_set_fill_style(ctx, 0xFF000000); // Black color
    
    // Set font (this would typically be done through the WASM wrapper)
    // In a real WASM implementation, you would use the JavaScript interop
    // to load actual fonts
    
    // For demonstration, we'll just show what the API calls would look like
    printf("Rendering text example...\n");
    
    // Fill text
    wcn_fill_text(ctx, "Hello, WCN WASM!", 100.0f, 100.0f);
    
    // Measure text
    WCN_TextMetrics metrics = wcn_measure_text(ctx, "Hello, WCN WASM!");
    printf("Text metrics - Width: %.2f\n", metrics.width);
    
    // You can also set different font properties
    wcn_set_font(ctx, "24px Arial");
    wcn_fill_text(ctx, "Larger text example", 100.0f, 150.0f);
    
    printf("Text rendering complete\n");
}

/**
 * @brief Example of loading a font through the WASM decoder
 * 
 * @param ctx WCN context
 * @param font_name Name of the font to load
 * @param font_size Size of the font
 */
void example_load_font(WCN_Context* ctx, const char* font_name, float font_size) {
    if (!ctx || !font_name) {
        printf("Invalid parameters\n");
        return;
    }
    
    printf("Loading font: %s at size %.1f\n", font_name, font_size);
    
    // In a real WASM implementation, this would involve:
    // 1. Passing the font name to JavaScript
    // 2. JavaScript would load the actual font file
    // 3. JavaScript would return a handle to the loaded font
    // 4. The C code would use that handle for rendering
    
    // For now, we'll just demonstrate the API structure
    printf("In a real WASM build, this would load the font through JavaScript interop\n");
    
    // Set the font for subsequent text operations
    char font_spec[256];
    snprintf(font_spec, sizeof(font_spec), "%.1fpx %s", font_size, font_name);
    wcn_set_font(ctx, font_spec);
    
    printf("Font set to: %s\n", font_spec);
}

// ============================================================================
// Exported functions for WASM
// ============================================================================

#ifdef __EMSCRIPTEN__

/**
 * @brief Initialize the WASM font decoder example
 * 
 * This function would be called from JavaScript to set up the example
 */
EMSCRIPTEN_KEEPALIVE
void init_wasm_font_example(WCN_Context* ctx) {
    printf("Initializing WASM font decoder example\n");
    example_wasm_font_decoder_usage(ctx);
}

/**
 * @brief Render a text example
 * 
 * This function would be called from JavaScript to render text
 */
EMSCRIPTEN_KEEPALIVE
void render_wasm_text_example(WCN_Context* ctx) {
    printf("Rendering WASM text example\n");
    render_text_example(ctx);
}

/**
 * @brief Load a font
 * 
 * This function would be called from JavaScript to load a font
 */
EMSCRIPTEN_KEEPALIVE
void load_wasm_font(WCN_Context* ctx, const char* font_name, float font_size) {
    printf("Loading WASM font: %s\n", font_name);
    example_load_font(ctx, font_name, font_size);
}

#endif // __EMSCRIPTEN__

/**
 * @brief Main example function
 * 
 * This is just for demonstration - in a real WASM build,
 * the actual initialization would be done through JavaScript interop
 */
int main(void) {
    printf("WCN WASM Font Decoder Example\n");
    printf("This example shows how to use the WASM font decoder\n");
    printf("In a real WASM build, this would be called from JavaScript\n");
    
    // Note: We can't actually create a context here without WebGPU,
    // but we can show the API usage
    
    return 0;
}