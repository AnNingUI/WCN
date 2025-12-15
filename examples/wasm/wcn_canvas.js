// WCN Canvas Wrapper for WebAssembly
// Provides an HTML5 Canvas2D-like API for WCN WebAssembly builds

/**
 * Create a WCN Canvas wrapper around an HTML5 canvas element
 * @param {string|HTMLCanvasElement} canvasElement - Canvas element or its ID
 * @param {Object} WCNModule - The loaded WCN WebAssembly module
 * @returns {Promise<WCNCanvas>} - Promise that resolves to the WCN Canvas wrapper
 */
async function createWCNCanvas(canvasElement, WCNModule) {
    // Get the canvas element
    const canvas = typeof canvasElement === 'string'
        ? document.getElementById(canvasElement)
        : canvasElement;

    if (!canvas) {
        throw new Error('Canvas element not found');
    }

    // Initialize WCNJS if not already initialized
    if (typeof window.WCNJS === 'undefined') {
        WCNModule._wcn_init_js();
    }

    // Get WebGPU adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('Failed to get WebGPU adapter');
    }

    const device = await adapter.requestDevice();
    if (!device) {
        throw new Error('Failed to get WebGPU device');
    }

    // Store the device in the module's preinitializedWebGPUDevice
    WCNModule.preinitializedWebGPUDevice = device;

    // Create WCN GPU resources using auto mode (Emscripten will manage the device)
    const gpuResources = WCNModule._wcn_wasm_create_gpu_resources_auto();

    // Create WCN Context
    const context = WCNModule._wcn_create_context(gpuResources);
    if (!context) {
        throw new Error('Failed to create WCN context');
    }

    // Configure canvas for WebGPU
    const contextGPUCtx = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    contextGPUCtx.configure({
        device: device,
        format: format,
        alphaMode: 'premultiplied'
    });

    // Set the surface format
    const formatEnum = format === 'bgra8unorm' ? 23 : 18; // WGPUTextureFormat_BGRA8Unorm or RGBA8Unorm
    WCNModule._wcn_set_surface_format(context, formatEnum);

    // Register the WASM font decoder
    try {
        const fontDecoder = WCNModule._wcn_wasm_get_font_decoder();
        if (fontDecoder) {
            WCNModule._wcn_register_font_decoder(context, fontDecoder);
            console.log('[WCN Canvas] Registered WASM font decoder');
        }
    } catch (error) {
        console.warn('[WCN Canvas] Failed to register font decoder:', error);
    }

    // Create and set a default font face
    try {
        const defaultFontFace = WCNModule._wcn_wasm_create_default_font_face();
        if (defaultFontFace) {
            WCNModule._wcn_set_font_face(context, defaultFontFace, 16.0);
        }
    } catch (error) {
        console.warn('[WCN Canvas] Failed to create default font face:', error);
    }

    // Register the WASM image decoder
    try {
        if (WCNModule._wcn_wasm_get_image_decoder) {
            const imageDecoder = WCNModule._wcn_wasm_get_image_decoder();
            if (imageDecoder) {
                WCNModule._wcn_register_image_decoder(context, imageDecoder);
                console.log('[WCN Canvas] Registered WASM image decoder');
            }
        }
    } catch (error) {
        console.warn('[WCN Canvas] Failed to register image decoder:', error);
    }

    // Create WCN Canvas wrapper
    const wcnCanvas = new WCNCanvas(WCNModule, context, canvas, contextGPUCtx, device, format);

    return wcnCanvas;
}

/**
 * WCN Canvas wrapper class that provides HTML5 Canvas2D-like API
 */
class WCNCanvas {
    /**
     * Create a WCN Canvas wrapper
     * @param {Object} WCNModule - The loaded WCN WebAssembly module
     * @param {WCN_Context*} context - The WCN context
     * @param {HTMLCanvasElement} canvas - The HTML canvas element
     * @param {GPUCanvasContext} canvasContext - The WebGPU canvas context
     * @param {GPUDevice} device - The WebGPU device
     * @param {string} format - The WebGPU texture format
     */
    constructor(WCNModule, context, canvas, canvasContext, device, format) {
        this.WCN = WCNModule;
        this.context = context;
        this.canvas = canvas;
        this.canvasContext = canvasContext;
        this.device = device;
        this.format = format;
        this.textureViews = new Map();
        this.lastWidth = canvas.width;
        this.lastHeight = canvas.height;
        this.loadedFontFaces = new Map(); // font name -> font face pointer
    }

    /**
     * Begin a new frame
     */
    beginFrame() {
        // Check if canvas size has changed and update renderer if needed
        if (this.canvas.width !== this.lastWidth || this.canvas.height !== this.lastHeight) {
            // Update the context with new dimensions
            this.WCN._wcn_begin_frame(
                this.context,
                this.canvas.width,
                this.canvas.height,
                this.format === 'bgra8unorm' ? 23 : 18 // WGPUTextureFormat_BGRA8Unorm or RGBA8Unorm
            );
            this.lastWidth = this.canvas.width;
            this.lastHeight = this.canvas.height;
        } else {
            // Just begin frame with current dimensions
            this.WCN._wcn_begin_frame(
                this.context,
                this.canvas.width,
                this.canvas.height,
                this.format === 'bgra8unorm' ? 23 : 18 // WGPUTextureFormat_BGRA8Unorm or RGBA8Unorm
            );
        }
    }

    /**
     * End the current frame
     */
    endFrame() {
        this.WCN._wcn_end_frame(this.context);
    }

    /**
     * Begin a render pass
     * @returns {Object|null} Render pass info or null if failed
     */
    beginRenderPass() {
        // Get current texture from canvas context
        const texture = this.canvasContext.getCurrentTexture();
        if (!texture) {
            console.error('Failed to get current texture');
            return null;
        }

        // Create texture view
        const textureView = texture.createView();
        if (!textureView) {
            console.error('Failed to create texture view');
            return null;
        }

        // Store texture view and get an ID
        // The storeWGPUTextureView function takes the view and returns an ID
        const textureViewId = window.WCNJS.storeWGPUTextureView(textureView);

        // Begin render pass using the WASM wrapper function
        const result = this.WCN._wcn_begin_render_pass(this.context, textureViewId);

        // Return only the ID and result
        // We don't store the actual textureView object to avoid holding references
        // The textureView will be managed by the JavaScript Map in WCNJS
        return {
            textureViewId: textureViewId,
            result: result
        };
    }

    /**
     * End a render pass
     * @param {number} textureViewId - The texture view ID returned by beginRenderPass
     */
    endRenderPass(textureViewId) {
        this.WCN._wcn_end_render_pass(this.context);

        // Note: The texture view is automatically freed in wcn_submit_commands
        // We don't need to manually free it here to avoid double-free issues
    }

    /**
     * Submit commands to the GPU
     */
    submitCommands() {
        this.WCN._wcn_submit_commands(this.context);
    }

    // ==================== 2D Drawing Methods ====================

    /**
     * Save the current drawing state
     */
    save() {
        this.WCN._wcn_save(this.context);
    }

    /**
     * Restore the most recently saved drawing state
     */
    restore() {
        this.WCN._wcn_restore(this.context);
    }

    /**
     * Clear a rectangular area
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} width - Width of rectangle
     * @param {number} height - Height of rectangle
     */
    clearRect(x, y, width, height) {
        this.WCN._wcn_clear_rect(this.context, x, y, width, height);
    }

    /**
     * Fill a rectangular area
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} width - Width of rectangle
     * @param {number} height - Height of rectangle
     */
    fillRect(x, y, width, height) {
        this.WCN._wcn_fill_rect(this.context, x, y, width, height);
    }

    /**
     * Stroke a rectangular area
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} width - Width of rectangle
     * @param {number} height - Height of rectangle
     */
    strokeRect(x, y, width, height) {
        this.WCN._wcn_stroke_rect(this.context, x, y, width, height);
    }

    // ==================== Path Methods ====================

    /**
     * Begin a new path
     */
    beginPath() {
        this.WCN._wcn_begin_path(this.context);
    }

    /**
     * Move the path to a new point
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     */
    moveTo(x, y) {
        this.WCN._wcn_move_to(this.context, x, y);
    }

    /**
     * Add a line to the path
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     */
    lineTo(x, y) {
        this.WCN._wcn_line_to(this.context, x, y);
    }

    /**
     * Add an arc to the path
     * @param {number} x - X coordinate of the arc's center
     * @param {number} y - Y coordinate of the arc's center
     * @param {number} radius - Arc radius
     * @param {number} startAngle - Starting angle in radians
     * @param {number} endAngle - Ending angle in radians
     * @param {boolean} anticlockwise - Whether the arc should be drawn anticlockwise
     */
    arc(x, y, radius, startAngle, endAngle, anticlockwise = false) {
        this.WCN._wcn_arc(this.context, x, y, radius, startAngle, endAngle, anticlockwise ? 1 : 0);
    }

    /**
     * Fill a circle
     * @param {number} x - X coordinate of the circle's center
     * @param {number} y - Y coordinate of the circle's center
     * @param {number} radius - Circle radius
     */
    fillCircle(x, y, radius) {
        wcnCanvas.beginPath();
        wcnCanvas.arc(x, y, radius, 0, Math.PI * 2);
        wcnCanvas.fill();
    }

    /**
     * Add a rectangle to the path
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} width - Width of rectangle
     * @param {number} height - Height of rectangle
     */
    rect(x, y, width, height) {
        this.WCN._wcn_rect(this.context, x, y, width, height);
    }

    /**
     * Close the current path
     */
    closePath() {
        this.WCN._wcn_close_path(this.context);
    }

    /**
     * Fill the current path
     */
    fill() {
        this.WCN._wcn_fill(this.context);
    }

    /**
     * Stroke the current path
     */
    stroke() {
        this.WCN._wcn_stroke(this.context);
    }

    // ==================== Style Methods ====================

    /**
     * Set the fill style
     * @param {string|number} style - Color as CSS string or ARGB number
     */
    setFillStyle(style) {
        const color = this._parseColor(style);
        this.WCN._wcn_set_fill_style(this.context, color);
    }

    /**
     * Set the stroke style
     * @param {string|number} style - Color as CSS string or ARGB number
     */
    setStrokeStyle(style) {
        const color = this._parseColor(style);
        this.WCN._wcn_set_stroke_style(this.context, color);
    }

    /**
     * Set the line width
     * @param {number} width - Line width
     */
    setLineWidth(width) {
        this.WCN._wcn_set_line_width(this.context, width);
    }

    /**
     * Set the line cap style
     * @param {string} cap - Line cap style ('butt', 'round', 'square')
     */
    setLineCap(cap) {
        let capValue;
        switch (cap) {
            case 'butt':
                capValue = this.WCN._wcn_wasm_line_cap_butt();
                break;
            case 'round':
                capValue = this.WCN._wcn_wasm_line_cap_round();
                break;
            case 'square':
                capValue = this.WCN._wcn_wasm_line_cap_square();
                break;
            default:
                capValue = this.WCN._wcn_wasm_line_cap_butt();
        }
        this.WCN._wcn_set_line_cap(this.context, capValue);
    }

    /**
     * Set the line join style
     * @param {string} join - Line join style ('miter', 'round', 'bevel')
     */
    setLineJoin(join) {
        let joinValue;
        switch (join) {
            case 'miter':
                joinValue = this.WCN._wcn_wasm_line_join_miter();
                break;
            case 'round':
                joinValue = this.WCN._wcn_wasm_line_join_round();
                break;
            case 'bevel':
                joinValue = this.WCN._wcn_wasm_line_join_bevel();
                break;
            default:
                joinValue = this.WCN._wcn_wasm_line_join_miter();
        }
        this.WCN._wcn_set_line_join(this.context, joinValue);
    }

    /**
     * Set the miter limit
     * @param {number} limit - Miter limit
     */
    setMiterLimit(limit) {
        this.WCN._wcn_set_miter_limit(this.context, limit);
    }

    /**
     * Set the global alpha
     * @param {number} alpha - Alpha value (0.0 to 1.0)
     */
    setGlobalAlpha(alpha) {
        this.WCN._wcn_set_global_alpha(this.context, alpha);
    }

    // ==================== Transform Methods ====================

    /**
     * Translate the canvas origin
     * @param {number} x - X translation
     * @param {number} y - Y translation
     */
    translate(x, y) {
        this.WCN._wcn_translate(this.context, x, y);
    }

    /**
     * Rotate the canvas
     * @param {number} angle - Rotation angle in radians
     */
    rotate(angle) {
        this.WCN._wcn_rotate(this.context, angle);
    }

    /**
     * Scale the canvas
     * @param {number} x - X scale factor
     * @param {number} y - Y scale factor
     */
    scale(x, y) {
        this.WCN._wcn_scale(this.context, x, y);
    }

    /**
     * Apply a transformation matrix
     * @param {number} a - Horizontal scaling
     * @param {number} b - Horizontal skewing
     * @param {number} c - Vertical skewing
     * @param {number} d - Vertical scaling
     * @param {number} e - Horizontal moving
     * @param {number} f - Vertical moving
     */
    transform(a, b, c, d, e, f) {
        this.WCN._wcn_transform(this.context, a, b, c, d, e, f);
    }

    /**
     * Reset and apply a transformation matrix
     * @param {number} a - Horizontal scaling
     * @param {number} b - Horizontal skewing
     * @param {number} c - Vertical skewing
     * @param {number} d - Vertical scaling
     * @param {number} e - Horizontal moving
     * @param {number} f - Vertical moving
     */
    setTransform(a, b, c, d, e, f) {
        this.WCN._wcn_set_transform(this.context, a, b, c, d, e, f);
    }

    /**
     * Reset the transformation to the identity matrix
     */
    resetTransform() {
        this.WCN._wcn_reset_transform(this.context);
    }

    // ==================== Text Methods ====================

    /**
     * Set the current font
     * @param {string} font - CSS font string (e.g., "16px Arial")
     */
    setFont(font) {
        // Allocate memory for the font string
        const fontPtr = this.WCN._wcn_wasm_malloc(font.length + 1);
        this.WCN.stringToUTF8(font, fontPtr, font.length + 1);
        
        // Set the font (updates size alignment/baseline)
        this.WCN._wcn_set_font(this.context, fontPtr);
        
        // Free the allocated memory
        this.WCN._wcn_wasm_free(fontPtr);

        const match = font.match(/^\s*([0-9]*\.?[0-9]+)\s*px\s*(.+)$/i);
        if (match) {
            const size = parseFloat(match[1]);
            let family = match[2].split(',')[0].trim();
            // remove quotes if present
            family = family.replace(/^['"]+|['"]+$/g, '');
            const entry = this.loadedFontFaces.get(family.toLowerCase());
            if (entry && entry.ptr) {
                this.WCN._wcn_set_font_face(this.context, entry.ptr, size);
            }
        }
    }

    /**
     * Set the current font face
     * @param {string} fontFace - Font face name
     * @param {number} fontSize - Font size in pixels
     */
    setFontFace(fontFace, fontSize) {
        // For WASM, we need to load the font first
        // Allocate memory for the font face string
        const fontFacePtr = this.WCN._wcn_wasm_malloc(fontFace.length + 1);
        this.WCN.stringToUTF8(fontFace, fontFacePtr, fontFace.length + 1);

        // Set the font face
        this.WCN._wcn_set_font_face(this.context, fontFacePtr, fontSize);

        // Free the allocated memory
        this.WCN._wcn_wasm_free(fontFacePtr);
    }

    /**
     * Load a font and set it as the current font
     * @param {string} fontName - Name of the font to load
     * @param {number} fontSize - Font size in pixels
     */
    loadFont(fontName, fontSize = 16, options = {}) {
        const setCurrent = options && Object.prototype.hasOwnProperty.call(options, 'setCurrent')
            ? !!options.setCurrent
            : true;
        const addFallback = options && !!options.addFallback;

        if (!this.WCN._wcn_wasm_load_font) {
            console.error('[WCN Canvas] wcn_wasm_load_font is not exported. Rebuild wcn_wasm with the updated CMake.');
            return false;
        }
        if (addFallback && !this.WCN._wcn_add_font_fallback) {
            console.error('[WCN Canvas] wcn_add_font_fallback is not exported. Rebuild wcn_wasm with the updated CMake.');
            return false;
        }

        // Create font data - in WASM implementation, this is just the font name as a null-terminated string
        const fontDataString = fontName + '\0';
        const fontDataLength = fontDataString.length;

        // Allocate memory for font data
        const fontDataPtr = this.WCN._wcn_wasm_malloc(fontDataLength);
        this.WCN.stringToUTF8(fontDataString, fontDataPtr, fontDataLength);

        // Create a pointer to hold the font face
        const fontFacePtrPtr = this.WCN._wcn_wasm_malloc(4); // 4 bytes for a pointer

        try {
            // Load the font using the WASM font decoder
            const fontDecoder = this.WCN._wcn_wasm_get_font_decoder();
            if (fontDecoder && this.WCN._wcn_wasm_load_font) {
                const result = this.WCN._wcn_wasm_load_font(fontDataPtr, fontDataLength, fontFacePtrPtr);
                if (result) {
                    // Get the font face pointer
                    const fontFacePtr = this.WCN.getValue(fontFacePtrPtr, 'i32');
                    if (fontFacePtr) {
                        if (addFallback) {
                            if (this.WCN._wcn_add_font_fallback) {
                                const added = this.WCN._wcn_add_font_fallback(this.context, fontFacePtr);
                                if (!added) {
                                    console.warn(`[WCN Canvas] Failed to register fallback font: ${fontName}`);
                                }
                            } else {
                                console.warn('[WCN Canvas] wcn_add_font_fallback is not exported in this build.');
                            }
                        }

                        // remember face for future setFont calls
                        this.loadedFontFaces.set(fontName.toLowerCase(), { ptr: fontFacePtr });

                        if (setCurrent) {
                            this.WCN._wcn_set_font_face(this.context, fontFacePtr, fontSize);
                        }

                        console.log(`[WCN Canvas] Loaded font: ${fontName}${addFallback ? ' (fallback)' : ''}`);
                        return true;
                    }
                }
            }
        } catch (error) {
            console.error('[WCN Canvas] Failed to load font:', error);
        } finally {
            // Free allocated memory
            this.WCN._wcn_wasm_free(fontDataPtr);
            this.WCN._wcn_wasm_free(fontFacePtrPtr);
        }

        console.warn(`[WCN Canvas] Unable to load font ${fontName}`);
        return false;
    }

    /**
     * Load a font and register it as a fallback without changing the current face.
     * @param {string} fontName
     * @param {number} fontSize
     */
    loadFallbackFont(fontName, fontSize = 16) {
        return this.loadFont(fontName, fontSize, { setCurrent: false, addFallback: true });
    }

    /**
     * Fill text
     * @param {string} text - Text to fill
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     */
    fillText(text, x, y) {
        // Allocate memory for the text string
        const textBytes = this.WCN.lengthBytesUTF8(text) + 1;
        const textPtr = this.WCN._wcn_wasm_malloc(textBytes);
        this.WCN.stringToUTF8(text, textPtr, textBytes);

        // Fill the text
        this.WCN._wcn_fill_text(this.context, textPtr, x, y);

        // Free the allocated memory
        this.WCN._wcn_wasm_free(textPtr);
    }

    /**
     * Stroke text
     * @param {string} text - Text to stroke
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     */
    strokeText(text, x, y) {
        // Allocate memory for the text string
        const textBytes = this.WCN.lengthBytesUTF8(text) + 1;
        const textPtr = this.WCN._wcn_wasm_malloc(textBytes);
        this.WCN.stringToUTF8(text, textPtr, textBytes);

        // Stroke the text
        this.WCN._wcn_stroke_text(this.context, textPtr, x, y);

        // Free the allocated memory
        this.WCN._wcn_wasm_free(textPtr);
    }

    /**
     * Measure text
     * @param {string} text - Text to measure
     * @returns {Object} Text metrics
     */
    measureText(text) {
        // Allocate memory for the text string
        const textBytes = this.WCN.lengthBytesUTF8(text) + 1;
        const textPtr = this.WCN._wcn_wasm_malloc(textBytes);
        this.WCN.stringToUTF8(text, textPtr, textBytes);

        // Measure the text
        const metrics = this.WCN._wcn_measure_text(this.context, textPtr);

        // Free the allocated memory
        this.WCN._wcn_wasm_free(textPtr);

        // Return a simplified metrics object
        return {
            width: metrics.width || 0
        };
    }

    /**
     * Create an image object from a raw byte array.
     * @param {Uint8Array} byteArray - Encoded image bytes.
     * @returns {WCNImage}
     */
    createImageFromBytes(byteArray) {
        if (!byteArray || !byteArray.length) {
            throw new Error('Byte array is empty');
        }
        if (!this.WCN._wcn_decode_image) {
            throw new Error('wcn_decode_image is not exported in the WASM module');
        }

        const dataPtr = this.WCN._wcn_wasm_malloc(byteArray.length);
        this.WCN.HEAPU8.set(byteArray, dataPtr);
        const imagePtr = this.WCN._wcn_decode_image(this.context, dataPtr, byteArray.length);
        this.WCN._wcn_wasm_free(dataPtr);

        if (!imagePtr) {
            throw new Error('Failed to decode image bytes');
        }

        return new WCNImage(this, imagePtr);
    }

    /**
     * Load an image from a URL.
     * @param {string} url - Image URL
     * @returns {Promise<WCNImage>}
     */
    async loadImageFromUrl(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.status}`);
        }
        const buffer = await response.arrayBuffer();
        return this.createImageFromBytes(new Uint8Array(buffer));
    }

    /**
     * Draw an image using Canvas2D-like semantics.
     * @param {WCNImage} image - Image object created via WCN.
     * @param {...number} params - Parameters following the HTML5 Canvas API.
     */
    drawImage(image, ...params) {
        if (!image || !image.ptr) {
            throw new Error('Invalid image passed to drawImage');
        }

        const count = params.length;
        if (count === 2) {
            const [dx, dy] = params;
            this.WCN._wcn_draw_image(this.context, image.ptr, dx, dy);
        } else if (count === 4) {
            const [dx, dy, dw, dh] = params;
            this.WCN._wcn_draw_image_scaled(this.context, image.ptr, dx, dy, dw, dh);
        } else if (count === 8) {
            const [sx, sy, sw, sh, dx, dy, dw, dh] = params;
            this.WCN._wcn_draw_image_source(this.context, image.ptr, sx, sy, sw, sh, dx, dy, dw, dh);
        } else {
            throw new Error(`drawImage expected 3, 5, or 9 arguments, got ${count + 1}`);
        }
    }

    // ==================== Private Helper Methods ====================

    /**
     * Parse a color string or number into ARGB format
     * @param {string|number} color - Color value
     * @returns {number} ARGB color as integer
     * @private
     */
    _parseColor(color) {
        if (typeof color === 'number') {
            return color;
        }

        if (typeof color !== 'string') {
            return 0xFF000000; // Default to black
        }

        // Handle common color formats
        if (color.startsWith('#')) {
            // Hex color
            const hex = color.substring(1);
            if (hex.length === 3) {
                // Short hex (#RGB)
                const r = parseInt(hex[0] + hex[0], 16);
                const g = parseInt(hex[1] + hex[1], 16);
                const b = parseInt(hex[2] + hex[2], 16);
                return 0xFF000000 | (r << 16) | (g << 8) | b;
            } else if (hex.length === 6) {
                // Full hex (#RRGGBB)
                const r = parseInt(hex.substring(0, 2), 16);
                const g = parseInt(hex.substring(2, 4), 16);
                const b = parseInt(hex.substring(4, 6), 16);
                return 0xFF000000 | (r << 16) | (g << 8) | b;
            } else if (hex.length === 8) {
                // Full hex with alpha (#AARRGGBB)
                const a = parseInt(hex.substring(0, 2), 16);
                const r = parseInt(hex.substring(2, 4), 16);
                const g = parseInt(hex.substring(4, 6), 16);
                const b = parseInt(hex.substring(6, 8), 16);
                return (a << 24) | (r << 16) | (g << 8) | b;
            }
        } else if (color.startsWith('rgb(')) {
            // RGB color
            const match = color.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/);
            if (match) {
                const r = parseInt(match[1], 10);
                const g = parseInt(match[2], 10);
                const b = parseInt(match[3], 10);
                return 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        } else if (color.startsWith('rgba(')) {
            // RGBA color
            const match = color.match(/rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)/);
            if (match) {
                const r = parseInt(match[1], 10);
                const g = parseInt(match[2], 10);
                const b = parseInt(match[3], 10);
                const a = Math.round(parseFloat(match[4]) * 255);
                return (a << 24) | (r << 16) | (g << 8) | b;
            }
        }

        // Default to black for unknown formats
        return 0xFF000000;
    }
}
/**
 * Simple wrapper around WCN_ImageData for JS usage.
 */
class WCNImage {
    constructor(canvas, imagePtr) {
        this.canvas = canvas;
        this.WCN = canvas.WCN;
        this.ptr = imagePtr;
        this.width = this.WCN.getValue(imagePtr + 4, 'i32') >>> 0;
        this.height = this.WCN.getValue(imagePtr + 8, 'i32') >>> 0;
    }

    destroy() {
        if (this.ptr) {
            this.WCN._wcn_destroy_image_data(this.ptr);
            this.ptr = 0;
        }
    }
}
const Size = {
    f32: 4,
    vec2: 8
}
class Vec2 {
    static wcnModule = (/** @type {WCNModule} */(null));
    /**
     * @param {WCNModule} wcnModule
     */
    static init(wcnModule) {
        Vec2.wcnModule = wcnModule;
    }
    // ptr -> [f32, f32]
    ptr
    /**
     * @param {number} x
     * @param {number} y
     * */
    static makeXY(x, y) {
        const ptr = Vec2.wcnModule._malloc(Size.vec2);
        Vec2.wcnModule._wcn_math_Vec2_create_wasm(ptr, x, y);
        const vec = new Vec2();
        vec.ptr = ptr;
        return vec;
    }
    /**
     * @param {Vec2} v
     * */
    static copy(v) {
        const ptr = Vec2.wcnModule._malloc(Size.vec2);
        Vec2.wcnModule._wcn_math_Vec2_copy_wasm(ptr, v.ptr);
        const vec = new Vec2();
        vec.ptr = ptr;
        return vec;
    }
    free() {
        Vec2.wcnModule._free(this.ptr);
    }
    /**
     * @param {Vec2} other
     * */
    add(other) {
        const ptr = Vec2.wcnModule._malloc(Size.vec2);
        Vec2.wcnModule._wcn_math_Vec2_add_wasm(ptr, this.ptr, other.ptr)
        const vec = new Vec2();
        vec.ptr = ptr;
        return vec;
    }
    toString() {
        return `[${Vec2.wcnModule.getValue(this.ptr, 'float')}, ${Vec2.wcnModule.getValue(this.ptr + Size.f32, 'float')}]`;
    }

    toF32Array() {
        return new Float32Array(Vec2.wcnModule.HEAPF32.buffer, Vec2.wcnModule.HEAPF32.byteOffset + this.ptr, 2);
    }

    /**
     * @returns {[number, number]}
     * */
    toArray() {
        return [Vec2.wcnModule.getValue(this.ptr, 'float'), Vec2.wcnModule.getValue(this.ptr + Size.f32, 'float')];
    }

    get x() {
        return Vec2.wcnModule.getValue(this.ptr, 'float');
    }

    get y() {
        return Vec2.wcnModule.getValue(this.ptr + Size.f32, 'float');
    }

    set x(value) {
        Vec2.wcnModule._wcn_math_Vec2_set_x_wasm(this.ptr, value);
    }
    set y(value) {
        Vec2.wcnModule._wcn_math_Vec2_set_y_wasm(this.ptr, value);
    }
}

// Export the createWCNCanvas function
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createWCNCanvas };
} else if (typeof window !== 'undefined') {
    window.createWCNCanvas = createWCNCanvas;
    window.WCNImage = WCNImage;
    window.Vec2 = Vec2;
}
