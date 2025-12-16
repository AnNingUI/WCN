#include "wcn_internal.h"
#include "WCN/WCN_WASM.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

// ============================================================================
// WCN WASM Font Decoder Implementation
// Worker + OffscreenCanvas 优化版本
// ============================================================================

typedef struct {
    char* font_name;
    float base_size;
    int js_id;
} WCN_WASM_FontData;

// ============================================================================
// JavaScript Interop - Worker + OffscreenCanvas 架构
// ============================================================================

#ifdef __EMSCRIPTEN__

// 初始化 Worker + OffscreenCanvas 系统
// Worker 用于后台预渲染，主线程用于同步渲染（带缓存）
EM_JS(void, js_ensure_context, (), {
    if (typeof window.WCNJS === 'undefined') {
        window.WCNJS = {};
    }
    if (window.WCNJS.initialized) return;
    
    // 主线程 Canvas (同步渲染回退)
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    window.WCNJS.canvas = canvas;
    window.WCNJS.ctx = canvas.getContext('2d', { willReadFrequently: true });
    window.WCNJS.fonts = {};
    window.WCNJS.nextFontId = 1;
    
    // LRU 缓存 (主线程和Worker共享结果)
    window.WCNJS.cache = new Map();
    window.WCNJS.CACHE_MAX = 1024;
    
    // Worker 用于后台预渲染 (二进制通信协议)
    // 请求: ArrayBuffer [cmd:u8, fontId:u32, size:f32, count:u16, cps:u32[]]
    // 响应: ArrayBuffer [fontId:u32, size:u32, count:u16, items:{cp:u32,w:u16,h:u16,offX:f32,offY:f32,adv:f32,isColor:u8,pixels}]
    const workerCode = `
        let canvas = null;
        let ctx = null;
        let fonts = {};
        let nextFontId = 1;
        
        function initCanvas(sz) {
            if (!canvas) {
                canvas = new OffscreenCanvas(sz || 512, sz || 512);
                ctx = canvas.getContext('2d', { willReadFrequently: true });
            } else if (canvas.width < sz) {
                canvas.width = canvas.height = sz;
                ctx = canvas.getContext('2d', { willReadFrequently: true });
            }
        }
        
        function genBitmap(fontId, cp, size) {
            const f = fonts[fontId];
            if (!f) return null;
            
            const ch = String.fromCodePoint(cp);
            const pad = 4, need = Math.ceil(size + pad * 2);
            initCanvas(need);
            
            ctx.font = size + 'px ' + f.name;
            ctx.textBaseline = 'alphabetic';
            ctx.textAlign = 'left';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#FFF';
            
            const dx = pad, dy = Math.round(size);
            ctx.fillText(ch, dx, dy);
            
            const m = ctx.measureText(ch);
            const scanW = Math.min(canvas.width, Math.ceil(dx + m.width + pad));
            const scanH = Math.min(canvas.height, Math.ceil(dy + (m.actualBoundingBoxDescent || size * 0.3) + pad));
            
            const img = ctx.getImageData(0, 0, scanW, scanH);
            const d = img.data;
            
            let minX = scanW, maxX = 0, minY = scanH, maxY = 0;
            let hasPixels = false, isColor = false;
            
            for (let y = 0; y < scanH; y++) {
                for (let x = 0; x < scanW; x++) {
                    const i = (y * scanW + x) * 4;
                    if (d[i + 3] > 0) {
                        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
                        hasPixels = true;
                        if (!isColor && (Math.abs(d[i]-d[i+1]) > 2 || Math.abs(d[i+1]-d[i+2]) > 2)) isColor = true;
                    }
                }
            }
            
            if (!hasPixels) return { w: 1, h: 1, offX: 0, offY: 0, adv: m.width, isColor: 0, buf: new Uint8Array(4) };
            
            minX = Math.max(0, minX - 1); maxX = Math.min(scanW - 1, maxX + 1);
            minY = Math.max(0, minY - 1); maxY = Math.min(scanH - 1, maxY + 1);
            
            const w = maxX - minX + 1, h = maxY - minY + 1;
            const buf = new Uint8Array(w * h * 4);
            
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const si = ((minY + y) * scanW + (minX + x)) * 4, di = (y * w + x) * 4;
                    if (isColor) {
                        buf[di] = d[si]; buf[di+1] = d[si+1]; buf[di+2] = d[si+2]; buf[di+3] = d[si+3];
                    } else {
                        const a = d[si + 3];
                        buf[di] = buf[di+1] = buf[di+2] = a; buf[di+3] = 255;
                    }
                }
            }
            return { w, h, offX: minX - dx, offY: minY - dy, adv: m.width, isColor: isColor ? 1 : 0, buf };
        }
        
        self.onmessage = function(e) {
            const data = e.data;
            
            // 二进制请求
            if (data instanceof ArrayBuffer) {
                const dv = new DataView(data);
                const cmd = dv.getUint8(0);
                
                if (cmd === 1) { // loadFont: [cmd:1, nameLen:u16, name..., size:f32]
                    const nameLen = dv.getUint16(1, true);
                    const name = new TextDecoder().decode(new Uint8Array(data, 3, nameLen));
                    const size = dv.getFloat32(3 + nameLen, true);
                    initCanvas(512);
                    const id = nextFontId++;
                    fonts[id] = { name, size };
                    // 响应: [cmd:1, fontId:u32]
                    const resp = new ArrayBuffer(5);
                    new DataView(resp).setUint8(0, 1);
                    new DataView(resp).setUint32(1, id, true);
                    self.postMessage(resp, [resp]);
                }
                else if (cmd === 2) { // prerender: [cmd:2, fontId:u32, size:f32, count:u16, cps:u32[]]
                    const fontId = dv.getUint32(1, true);
                    const size = dv.getFloat32(5, true);
                    const count = dv.getUint16(9, true);
                    
                    // 渲染所有字形
                    const items = [];
                    let totalPixels = 0;
                    for (let i = 0; i < count; i++) {
                        const cp = dv.getUint32(11 + i * 4, true);
                        const r = genBitmap(fontId, cp, size);
                        if (r) {
                            items.push({ cp, ...r });
                            totalPixels += r.buf.length;
                        }
                    }
                    
                    // 打包二进制响应: [cmd:2, fontId:u32, size:u32, count:u16, items...]
                    // item: [cp:u32, w:u16, h:u16, offX:f32, offY:f32, adv:f32, isColor:u8, pixels...]
                    const headerSize = 12; // cmd + fontId + size + count
                    const itemHeaderSize = 21; // cp + w + h + offX + offY + adv + isColor
                    let respSize = headerSize;
                    for (const it of items) respSize += itemHeaderSize + it.buf.length;
                    
                    const resp = new ArrayBuffer(respSize);
                    const rdv = new DataView(resp);
                    const ru8 = new Uint8Array(resp);
                    
                    rdv.setUint8(0, 2);
                    rdv.setUint32(1, fontId, true);
                    rdv.setUint32(5, size, true);
                    rdv.setUint16(9, items.length, true);
                    
                    let offset = headerSize;
                    for (const it of items) {
                        rdv.setUint32(offset, it.cp, true);
                        rdv.setUint16(offset + 4, it.w, true);
                        rdv.setUint16(offset + 6, it.h, true);
                        rdv.setFloat32(offset + 8, it.offX, true);
                        rdv.setFloat32(offset + 12, it.offY, true);
                        rdv.setFloat32(offset + 16, it.adv, true);
                        rdv.setUint8(offset + 20, it.isColor);
                        ru8.set(it.buf, offset + 21);
                        offset += itemHeaderSize + it.buf.length;
                    }
                    
                    self.postMessage(resp, [resp]);
                }
                return;
            }
        };
    `;
    
    try {
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const worker = new Worker(URL.createObjectURL(blob));
        window.WCNJS.worker = worker;
        
        worker.onmessage = function(e) {
            const data = e.data;
            if (!(data instanceof ArrayBuffer)) return;
            
            const dv = new DataView(data);
            const cmd = dv.getUint8(0);
            
            if (cmd === 2) { // prerendered 响应
                const fontId = dv.getUint32(1, true);
                const size = dv.getUint32(5, true);
                const count = dv.getUint16(9, true);
                
                let offset = 12;
                for (let i = 0; i < count; i++) {
                    const cp = dv.getUint32(offset, true);
                    const w = dv.getUint16(offset + 4, true);
                    const h = dv.getUint16(offset + 6, true);
                    const offX = dv.getFloat32(offset + 8, true);
                    const offY = dv.getFloat32(offset + 12, true);
                    const adv = dv.getFloat32(offset + 16, true);
                    const isColor = dv.getUint8(offset + 20);
                    const pixelLen = w * h * 4;
                    
                    const key = fontId + '_' + cp + '_' + size;
                    
                    // LRU 缓存
                    if (window.WCNJS.cache.size >= window.WCNJS.CACHE_MAX) {
                        const oldest = window.WCNJS.cache.keys().next().value;
                        const old = window.WCNJS.cache.get(oldest);
                        if (old && old.ptr) Module._free(old.ptr);
                        window.WCNJS.cache.delete(oldest);
                    }
                    
                    const ptr = Module._malloc(pixelLen);
                    if (ptr) {
                        Module.HEAPU8.set(new Uint8Array(data, offset + 21, pixelLen), ptr);
                        window.WCNJS.cache.set(key, { ptr, w, h, offX, offY, adv, isColor });
                    }
                    
                    offset += 21 + pixelLen;
                }
            }
        };
    } catch (e) {
        console.warn('[WCN] Worker creation failed, using main thread only');
    }
    
    window.WCNJS.initialized = true;
});

// 同步加载字体
EM_JS(bool, js_load_font, (const char* font_name, float font_size, int* out_id), {
    try {
        js_ensure_context();
        const nameStr = UTF8ToString(font_name);
        const id = window.WCNJS.nextFontId++;
        window.WCNJS.fonts[id] = { name: nameStr, size: font_size };
        setValue(out_id, id, 'i32');
        
        // 通知 Worker 加载字体 (二进制协议)
        if (window.WCNJS.worker) {
            const nameBytes = new TextEncoder().encode(nameStr);
            const buf = new ArrayBuffer(3 + nameBytes.length + 4);
            const dv = new DataView(buf);
            dv.setUint8(0, 1); // cmd=loadFont
            dv.setUint16(1, nameBytes.length, true);
            new Uint8Array(buf, 3, nameBytes.length).set(nameBytes);
            dv.setFloat32(3 + nameBytes.length, font_size, true);
            window.WCNJS.worker.postMessage(buf, [buf]);
        }
        
        return true;
    } catch (e) {
        console.error("[WCN] Load font failed:", e);
        return false;
    }
});

// 预渲染字符串中的所有字形 (后台Worker执行，二进制协议)
EM_JS(void, js_prerender_text, (int font_id, const char* text, float size), {
    if (!window.WCNJS.worker) return;
    
    const str = UTF8ToString(text);
    const codepoints = [];
    for (const ch of str) {
        const cp = ch.codePointAt(0);
        const key = font_id + '_' + cp + '_' + (size|0);
        if (!window.WCNJS.cache.has(key)) {
            codepoints.push(cp);
        }
    }
    
    if (codepoints.length > 0) {
        // 二进制请求: [cmd:2, fontId:u32, size:f32, count:u16, cps:u32[]]
        const buf = new ArrayBuffer(11 + codepoints.length * 4);
        const dv = new DataView(buf);
        dv.setUint8(0, 2);
        dv.setUint32(1, font_id, true);
        dv.setFloat32(5, size, true);
        dv.setUint16(9, codepoints.length, true);
        for (let i = 0; i < codepoints.length; i++) {
            dv.setUint32(11 + i * 4, codepoints[i], true);
        }
        window.WCNJS.worker.postMessage(buf, [buf]);
    }
});

// 预渲染常用字符 (ASCII + 中文标点，二进制协议)
EM_JS(void, js_prerender_common, (int font_id, float size), {
    if (!window.WCNJS.worker) return;
    
    const codepoints = [];
    // ASCII 可打印字符
    for (let cp = 32; cp < 127; cp++) {
        const key = font_id + '_' + cp + '_' + (size|0);
        if (!window.WCNJS.cache.has(key)) codepoints.push(cp);
    }
    // 常用中文标点 (使用 Unicode 码点避免编码问题)
    // ，。！？、；：""''（）【】《》—…
    const commonCps = [0xFF0C, 0x3002, 0xFF01, 0xFF1F, 0x3001, 0xFF1B, 0xFF1A,
                       0x201C, 0x201D, 0x2018, 0x2019, 0xFF08, 0xFF09,
                       0x3010, 0x3011, 0x300A, 0x300B, 0x2014, 0x2026];
    for (let i = 0; i < commonCps.length; i++) {
        const cp = commonCps[i];
        const key = font_id + '_' + cp + '_' + (size|0);
        if (!window.WCNJS.cache.has(key)) codepoints.push(cp);
    }
    
    if (codepoints.length > 0) {
        // 二进制请求: [cmd:2, fontId:u32, size:f32, count:u16, cps:u32[]]
        const buf = new ArrayBuffer(11 + codepoints.length * 4);
        const dv = new DataView(buf);
        dv.setUint8(0, 2);
        dv.setUint32(1, font_id, true);
        dv.setFloat32(5, size, true);
        dv.setUint16(9, codepoints.length, true);
        for (let i = 0; i < codepoints.length; i++) {
            dv.setUint32(11 + i * 4, codepoints[i], true);
        }
        window.WCNJS.worker.postMessage(buf, [buf]);
    }
});


// 同步获取字形度量 (主线程)
EM_JS(bool, js_get_glyph_metrics, (int font_id, uint32_t codepoint,
                                  float* out_advance, float* out_lsb, float* out_box), {
    try {
        const font = window.WCNJS.fonts[font_id];
        if (!font) return false;

        const ctx = window.WCNJS.ctx;
        ctx.font = font.size + 'px ' + font.name;
        ctx.textBaseline = 'alphabetic';

        const charStr = String.fromCodePoint(codepoint);
        const m = ctx.measureText(charStr);

        setValue(out_advance, m.width, 'float');
        const lsb = m.actualBoundingBoxLeft ? -m.actualBoundingBoxLeft : 0;
        setValue(out_lsb, lsb, 'float');

        setValue(out_box + 0, lsb, 'float');
        setValue(out_box + 4, m.actualBoundingBoxAscent ? -m.actualBoundingBoxAscent : -font.size, 'float');
        setValue(out_box + 8, m.actualBoundingBoxRight || m.width, 'float');
        setValue(out_box + 12, m.actualBoundingBoxDescent || 0, 'float');

        return true;
    } catch (e) {
        return false;
    }
});

// 同步生成位图 (带 LRU 缓存)
EM_JS(bool, js_generate_bitmap, (int font_id, uint32_t codepoint, float size,
                                unsigned char** out_ptr, int* out_w, int* out_h,
                                float* out_off_x, float* out_off_y, float* out_adv,
                                bool* out_is_color_ptr), {
    try {
        const font = window.WCNJS.fonts[font_id];
        if (!font) return false;
        
        // LRU 缓存检查
        const cacheKey = font_id + '_' + codepoint + '_' + (size|0);
        const cached = window.WCNJS.cache.get(cacheKey);
        if (cached) {
            window.WCNJS.cache.delete(cacheKey);
            window.WCNJS.cache.set(cacheKey, cached);
            
            const bufSize = cached.w * cached.h * 4;
            const ptr = Module._malloc(bufSize);
            if (!ptr) return false;
            Module.HEAPU8.copyWithin(ptr, cached.ptr, cached.ptr + bufSize);
            
            setValue(out_ptr, ptr, 'i32');
            setValue(out_w, cached.w, 'i32');
            setValue(out_h, cached.h, 'i32');
            setValue(out_off_x, cached.offX, 'float');
            setValue(out_off_y, cached.offY, 'float');
            setValue(out_adv, cached.adv, 'float');
            if (out_is_color_ptr) setValue(out_is_color_ptr, cached.isColor, 'i8');
            return true;
        }

        const canvas = window.WCNJS.canvas;
        const charStr = String.fromCodePoint(codepoint);
        const padding = 4;
        const neededSize = Math.ceil(size + padding * 2);
        
        if (canvas.width < neededSize || canvas.height < neededSize) {
            canvas.width = Math.max(canvas.width, neededSize);
            canvas.height = Math.max(canvas.height, neededSize);
            window.WCNJS.ctx = canvas.getContext('2d', { willReadFrequently: true });
        }

        const ctx = window.WCNJS.ctx;
        ctx.font = size + 'px ' + font.name;
        ctx.textBaseline = 'alphabetic';
        ctx.textAlign = 'left';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#FFFFFF';

        const drawX = padding;
        const drawY = Math.round(size);
        ctx.fillText(charStr, drawX, drawY);
        
        const metrics = ctx.measureText(charStr);
        const scanW = Math.min(canvas.width, Math.ceil(drawX + metrics.width + padding));
        const scanH = Math.min(canvas.height, Math.ceil(drawY + (metrics.actualBoundingBoxDescent || size * 0.3) + padding));

        const imgData = ctx.getImageData(0, 0, scanW, scanH);
        const data = imgData.data;

        let minX = scanW, maxX = 0, minY = scanH, maxY = 0;
        let hasPixels = false, isColor = false;

        for (let y = 0; y < scanH; y++) {
            for (let x = 0; x < scanW; x++) {
                const idx = (y * scanW + x) * 4;
                if (data[idx + 3] > 0) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    hasPixels = true;
                    if (!isColor) {
                        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                        if (Math.abs(r - g) > 2 || Math.abs(g - b) > 2) isColor = true;
                    }
                }
            }
        }

        if (!hasPixels) {
            // 空字符
            const ptr = Module._malloc(4);
            if (!ptr) return false;
            Module.HEAPU8.fill(0, ptr, ptr + 4);
            setValue(out_ptr, ptr, 'i32');
            setValue(out_w, 1, 'i32');
            setValue(out_h, 1, 'i32');
            setValue(out_off_x, 0, 'float');
            setValue(out_off_y, 0, 'float');
            setValue(out_adv, metrics.width, 'float');
            if (out_is_color_ptr) setValue(out_is_color_ptr, false, 'i8');
            return true;
        }
        
        minX = Math.max(0, minX - 1);
        maxX = Math.min(scanW - 1, maxX + 1);
        minY = Math.max(0, minY - 1);
        maxY = Math.min(scanH - 1, maxY + 1);

        const w = maxX - minX + 1;
        const h = maxY - minY + 1;
        const bufSize = w * h * 4;
        const ptr = Module._malloc(bufSize);
        if (!ptr) return false;

        const heap = Module.HEAPU8;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const srcIdx = ((minY + y) * scanW + (minX + x)) * 4;
                const dstIdx = ptr + (y * w + x) * 4;
                if (isColor) {
                    heap[dstIdx] = data[srcIdx];
                    heap[dstIdx + 1] = data[srcIdx + 1];
                    heap[dstIdx + 2] = data[srcIdx + 2];
                    heap[dstIdx + 3] = data[srcIdx + 3];
                } else {
                    const alpha = data[srcIdx + 3];
                    heap[dstIdx] = alpha;
                    heap[dstIdx + 1] = alpha;
                    heap[dstIdx + 2] = alpha;
                    heap[dstIdx + 3] = 255;
                }
            }
        }
        
        const offX = minX - drawX;
        const offY = minY - drawY;

        // LRU 缓存存入
        if (window.WCNJS.cache.size >= window.WCNJS.CACHE_MAX) {
            const oldest = window.WCNJS.cache.keys().next().value;
            const oldEntry = window.WCNJS.cache.get(oldest);
            if (oldEntry && oldEntry.ptr) Module._free(oldEntry.ptr);
            window.WCNJS.cache.delete(oldest);
        }
        const cachePtr = Module._malloc(bufSize);
        if (cachePtr) {
            Module.HEAPU8.copyWithin(cachePtr, ptr, ptr + bufSize);
            window.WCNJS.cache.set(cacheKey, { ptr: cachePtr, w, h, offX, offY, adv: metrics.width, isColor });
        }

        setValue(out_ptr, ptr, 'i32');
        setValue(out_w, w, 'i32');
        setValue(out_h, h, 'i32');
        setValue(out_off_x, offX, 'float');
        setValue(out_off_y, offY, 'float');
        setValue(out_adv, metrics.width, 'float');
        if (out_is_color_ptr) setValue(out_is_color_ptr, isColor, 'i8');

        return true;
    } catch (e) {
        console.error("Bitmap gen failed", e);
        return false;
    }
});

#endif

// ============================================================================
// C Implementation
// ============================================================================

WCN_WASM_EXPORT bool wcn_wasm_load_font(const void* font_data, size_t data_size, WCN_FontFace** out_face) {
    if (!font_data || !out_face) return false;

    const char* font_name = (const char*)font_data;

    WCN_WASM_FontData* priv = (WCN_WASM_FontData*)malloc(sizeof(WCN_WASM_FontData));
    if (!priv) return false;

    priv->font_name = strdup(font_name);
    priv->base_size = 16.0f;
    priv->js_id = 0;

#ifdef __EMSCRIPTEN__
    if (!js_load_font(font_name, priv->base_size, &priv->js_id)) {
        free(priv->font_name);
        free(priv);
        return false;
    }
#endif

    WCN_FontFace* face = (WCN_FontFace*)malloc(sizeof(WCN_FontFace));
    if (!face) {
        free(priv->font_name);
        free(priv);
        return false;
    }

    face->family_name = priv->font_name;
    face->ascent = 800.0f;
    face->descent = -200.0f;
    face->line_gap = 100.0f;
    face->units_per_em = 1000.0f;
    face->user_data = priv;

    *out_face = face;
    return true;
}

static bool wcn_wasm_get_glyph(WCN_FontFace* face, uint32_t codepoint, WCN_Glyph** out_glyph) {
    if (!face || !out_glyph) return false;

    WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;

    WCN_Glyph* glyph = (WCN_Glyph*)malloc(sizeof(WCN_Glyph));
    if (!glyph) return false;

    memset(glyph, 0, sizeof(WCN_Glyph));
    glyph->codepoint = codepoint;

#ifdef __EMSCRIPTEN__
    float advance = 0, lsb = 0;
    float box[4] = {0};

    if (js_get_glyph_metrics(data->js_id, codepoint, &advance, &lsb, box)) {
        float scale = face->units_per_em / data->base_size;
        glyph->advance_width = advance * scale;
        glyph->left_side_bearing = lsb * scale;
        glyph->bounding_box[0] = box[0] * scale;
        glyph->bounding_box[1] = box[1] * scale;
        glyph->bounding_box[2] = box[2] * scale;
        glyph->bounding_box[3] = box[3] * scale;
    }
#else
    glyph->advance_width = 500.0f;
#endif

    glyph->contours = NULL;
    glyph->contour_count = 0;

    *out_glyph = glyph;
    return true;
}


// 同步获取字形 SDF 位图
static bool wcn_wasm_get_glyph_sdf(WCN_FontFace* face, uint32_t codepoint, float font_size,
                                  unsigned char** out_bitmap,
                                  int* out_width, int* out_height,
                                  float* out_offset_x, float* out_offset_y,
                                  float* out_advance,
                                  bool* out_is_color) {
    if (!face || !out_bitmap || !out_width || !out_height) {
        return false;
    }
    
    if (out_is_color) *out_is_color = false;

    WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;

#ifdef __EMSCRIPTEN__
    return js_generate_bitmap(font_data->js_id, codepoint, font_size,
                             out_bitmap, out_width, out_height,
                             out_offset_x, out_offset_y, out_advance,
                             out_is_color);
#else
    return false;
#endif
}

static void wcn_wasm_free_glyph_sdf(unsigned char* bitmap) {
    if (bitmap) {
        free(bitmap);
    }
}

static bool wcn_wasm_measure_text(WCN_FontFace* face, const char* text, float font_size,
                                 float* out_width, float* out_height) {
    if (!face || !text) return false;

#ifdef __EMSCRIPTEN__
    *out_width = 0;
    *out_height = font_size;
    return true;
#else
    *out_width = (float)strlen(text) * font_size * 0.5f;
    *out_height = font_size;
    return true;
#endif
}

static void wcn_wasm_free_glyph(WCN_Glyph* glyph) {
    if (glyph) {
        free(glyph);
    }
}

static void wcn_wasm_free_font(WCN_FontFace* face) {
    if (face) {
        WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;
        if (data) {
            if (data->font_name) free(data->font_name);
            free(data);
        }
        free(face);
    }
}

// ============================================================================
// Global Instance
// ============================================================================

static WCN_FontDecoder wcn_wasm_decoder = {
    .load_font = wcn_wasm_load_font,
    .get_glyph = wcn_wasm_get_glyph,
    .get_glyph_sdf = wcn_wasm_get_glyph_sdf,
    .free_glyph_sdf = wcn_wasm_free_glyph_sdf,
    .measure_text = wcn_wasm_measure_text,
    .free_glyph = wcn_wasm_free_glyph,
    .free_font = wcn_wasm_free_font,
    .name = "wasm_canvas_decoder"
};

WCN_FontDecoder* wcn_get_wasm_font_decoder(void) {
    return &wcn_wasm_decoder;
}

#ifdef __EMSCRIPTEN__
WCN_WASM_EXPORT WCN_FontDecoder* wcn_wasm_get_font_decoder(void) {
    return wcn_get_wasm_font_decoder();
}

WCN_WASM_EXPORT WCN_FontFace* wcn_wasm_create_default_font_face(void) {
    WCN_FontFace* face = NULL;
    wcn_wasm_load_font("Arial", 5, &face);
    return face;
}

// 预渲染文本中的字形 (Worker后台执行，不阻塞)
WCN_WASM_EXPORT void wcn_wasm_prerender_text(WCN_FontFace* face, const char* text, float font_size) {
    if (!face || !text) return;
    WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;
    js_prerender_text(data->js_id, text, font_size);
}

// 预渲染常用字符 (Worker后台执行)
WCN_WASM_EXPORT void wcn_wasm_prerender_common(WCN_FontFace* face, float font_size) {
    if (!face) return;
    WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;
    js_prerender_common(data->js_id, font_size);
}
#endif
