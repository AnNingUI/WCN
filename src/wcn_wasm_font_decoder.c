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

// 初始化 Worker + OffscreenCanvas 系统 (二进制通信协议)
EM_JS(void, js_ensure_context, (), {
    if (typeof window.WCNJS === 'undefined') {
        window.WCNJS = {};
    }
    
    if (window.WCNJS.fontWorker) return;
    
    // Worker 代码 - 二进制通信协议
    // 请求格式: ArrayBuffer [cmd:u8, id:u32, ...params]
    // 响应格式: ArrayBuffer [id:u32, status:u8, ...data]
    // CMD: 1=loadFont, 2=genBitmap, 3=genBatch, 4=preload, 5=clearCache
    const workerCode = `
        let canvas = null;
        let ctx = null;
        let fonts = {};
        let nextFontId = 1;
        
        const CACHE_MAX = 512;
        let cache = new Map();
        
        function cacheKey(fontId, cp, size) { return fontId + '_' + cp + '_' + (size|0); }
        function cacheGet(key) {
            const v = cache.get(key);
            if (v) { cache.delete(key); cache.set(key, v); }
            return v;
        }
        function cacheSet(key, val) {
            if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value);
            cache.set(key, val);
        }
        
        function initCanvas(sz) {
            if (!canvas) {
                canvas = new OffscreenCanvas(sz || 512, sz || 512);
                ctx = canvas.getContext('2d', { willReadFrequently: true });
            } else if (canvas.width < sz) {
                canvas.width = canvas.height = sz;
                ctx = canvas.getContext('2d', { willReadFrequently: true });
            }
        }
        
        function genBitmapCore(fontId, cp, size) {
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
        
        function genBitmap(fontId, cp, size) {
            const key = cacheKey(fontId, cp, size);
            let c = cacheGet(key);
            if (c) return { ...c, buf: new Uint8Array(c.buf) };
            const r = genBitmapCore(fontId, cp, size);
            if (r) cacheSet(key, { ...r, buf: new Uint8Array(r.buf) });
            return r;
        }
        
        // 打包单个位图结果为二进制: [id:u32, status:u8, w:u16, h:u16, offX:f32, offY:f32, adv:f32, isColor:u8, pixels...]
        // Header: 4 + 1 + 2 + 2 + 4 + 4 + 4 + 1 = 22 bytes
        function packBitmapResult(id, r) {
            if (!r) {
                const buf = new ArrayBuffer(5);
                const dv = new DataView(buf);
                dv.setUint32(0, id, true);
                dv.setUint8(4, 0); // status=fail
                return buf;
            }
            const hdrSize = 22;
            const pixelSize = r.buf.length;
            const buf = new ArrayBuffer(hdrSize + pixelSize);
            const dv = new DataView(buf);
            const u8 = new Uint8Array(buf);
            
            dv.setUint32(0, id, true);
            dv.setUint8(4, 1); // status=ok
            dv.setUint16(5, r.w, true);
            dv.setUint16(7, r.h, true);
            dv.setFloat32(9, r.offX, true);
            dv.setFloat32(13, r.offY, true);
            dv.setFloat32(17, r.adv, true);
            dv.setUint8(21, r.isColor);
            u8.set(r.buf, hdrSize);
            return buf;
        }
        
        // 批量打包: [id:u32, count:u16, {w:u16, h:u16, offX:f32, offY:f32, adv:f32, isColor:u8, pixelLen:u32, pixels...}...]
        function packBatchResult(id, results) {
            let totalSize = 6; // id + count
            const itemHeaders = 19; // w+h+offX+offY+adv+isColor+pixelLen = 2+2+4+4+4+1+4
            for (const r of results) {
                totalSize += itemHeaders + (r ? r.buf.length : 0);
            }
            
            const buf = new ArrayBuffer(totalSize);
            const dv = new DataView(buf);
            const u8 = new Uint8Array(buf);
            
            dv.setUint32(0, id, true);
            dv.setUint16(4, results.length, true);
            
            let offset = 6;
            for (const r of results) {
                if (r) {
                    dv.setUint16(offset, r.w, true);
                    dv.setUint16(offset + 2, r.h, true);
                    dv.setFloat32(offset + 4, r.offX, true);
                    dv.setFloat32(offset + 8, r.offY, true);
                    dv.setFloat32(offset + 12, r.adv, true);
                    dv.setUint8(offset + 16, r.isColor);
                    dv.setUint32(offset + 17, r.buf.length, true);
                    u8.set(r.buf, offset + 21);
                    offset += itemHeaders + r.buf.length;
                } else {
                    dv.setUint16(offset, 0, true);
                    dv.setUint16(offset + 2, 0, true);
                    dv.setFloat32(offset + 4, 0, true);
                    dv.setFloat32(offset + 8, 0, true);
                    dv.setFloat32(offset + 12, 0, true);
                    dv.setUint8(offset + 16, 0);
                    dv.setUint32(offset + 17, 0, true);
                    offset += itemHeaders;
                }
            }
            return buf;
        }
        
        self.onmessage = function(e) {
            const data = e.data;
            
            // 二进制请求
            if (data instanceof ArrayBuffer) {
                const dv = new DataView(data);
                const cmd = dv.getUint8(0);
                const id = dv.getUint32(1, true);
                
                if (cmd === 1) { // loadFont: [cmd, id, nameLen:u16, name..., size:f32]
                    const nameLen = dv.getUint16(5, true);
                    const nameBytes = new Uint8Array(data, 7, nameLen);
                    const name = new TextDecoder().decode(nameBytes);
                    const size = dv.getFloat32(7 + nameLen, true);
                    
                    initCanvas(512);
                    const fontId = nextFontId++;
                    fonts[fontId] = { name, size };
                    
                    const resp = new ArrayBuffer(9);
                    const rdv = new DataView(resp);
                    rdv.setUint32(0, id, true);
                    rdv.setUint8(4, 1);
                    rdv.setUint32(5, fontId, true);
                    self.postMessage(resp, [resp]);
                }
                else if (cmd === 2) { // genBitmap: [cmd, id, fontId:u32, cp:u32, size:f32]
                    const fontId = dv.getUint32(5, true);
                    const cp = dv.getUint32(9, true);
                    const size = dv.getFloat32(13, true);
                    const r = genBitmap(fontId, cp, size);
                    const resp = packBitmapResult(id, r);
                    self.postMessage(resp, [resp]);
                }
                else if (cmd === 3) { // genBatch: [cmd, id, fontId:u32, size:f32, count:u16, cps:u32[]]
                    const fontId = dv.getUint32(5, true);
                    const size = dv.getFloat32(9, true);
                    const count = dv.getUint16(13, true);
                    const results = [];
                    for (let i = 0; i < count; i++) {
                        const cp = dv.getUint32(15 + i * 4, true);
                        results.push(genBitmap(fontId, cp, size));
                    }
                    const resp = packBatchResult(id, results);
                    self.postMessage(resp, [resp]);
                }
                else if (cmd === 4) { // preload: [cmd, id, fontId:u32, size:f32]
                    const fontId = dv.getUint32(5, true);
                    const size = dv.getFloat32(9, true);
                    for (let cp = 32; cp < 127; cp++) genBitmap(fontId, cp, size);
                    const common = '，。！？、；：""''（）【】《》—…';
                    for (const ch of common) genBitmap(fontId, ch.codePointAt(0), size);
                    const resp = new ArrayBuffer(5);
                    new DataView(resp).setUint32(0, id, true);
                    new DataView(resp).setUint8(4, 1);
                    self.postMessage(resp, [resp]);
                }
                else if (cmd === 5) { // clearCache
                    cache.clear();
                    const resp = new ArrayBuffer(5);
                    new DataView(resp).setUint32(0, id, true);
                    new DataView(resp).setUint8(4, 1);
                    self.postMessage(resp, [resp]);
                }
                return;
            }
            
            // 兼容旧的 JSON 格式 (loadFont 字符串传递)
            const { id, cmd, args } = data;
            if (cmd === 'loadFont') {
                initCanvas(512);
                const fontId = nextFontId++;
                fonts[fontId] = { name: args.name, size: args.size };
                self.postMessage({ id, result: fontId });
            }
        };
    `;
    
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const worker = new Worker(URL.createObjectURL(blob));
    
    window.WCNJS.fontWorker = worker;
    window.WCNJS.pendingCalls = {};
    window.WCNJS.callId = 0;
    window.WCNJS.fonts = {};
    window.WCNJS.nextFontId = 1;
    
    window.WCNJS.mainCache = new Map();
    window.WCNJS.MAIN_CACHE_MAX = 256;
    
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 512;
    window.WCNJS.canvas = canvas;
    window.WCNJS.ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    worker.onmessage = function(e) {
        const data = e.data;
        
        // 二进制响应
        if (data instanceof ArrayBuffer) {
            const dv = new DataView(data);
            const id = dv.getUint32(0, true);
            const cb = window.WCNJS.pendingCalls[id];
            if (cb) {
                delete window.WCNJS.pendingCalls[id];
                cb(data);
            }
            return;
        }
        
        // JSON 响应
        const { id, result } = data;
        const cb = window.WCNJS.pendingCalls[id];
        if (cb) {
            delete window.WCNJS.pendingCalls[id];
            cb(result);
        }
    };
});

// 异步调用 Worker
EM_JS(void, js_worker_call_async, (const char* cmd, const char* argsJson, int callbackId), {
    const cmdStr = UTF8ToString(cmd);
    const args = JSON.parse(UTF8ToString(argsJson));
    const id = window.WCNJS.callId++;
    
    window.WCNJS.pendingCalls[id] = function(result) {
        // 存储结果供 C 端轮询
        if (!window.WCNJS.results) window.WCNJS.results = {};
        window.WCNJS.results[callbackId] = result;
    };
    
    window.WCNJS.fontWorker.postMessage({ id, cmd: cmdStr, args });
});

// 检查异步结果是否就绪
EM_JS(bool, js_check_async_result, (int callbackId), {
    return window.WCNJS.results && window.WCNJS.results[callbackId] !== undefined;
});

// 同步加载字体 (主线程回退)
EM_JS(bool, js_load_font, (const char* font_name, float font_size, int* out_id), {
    try {
        js_ensure_context();
        const nameStr = UTF8ToString(font_name);
        const id = window.WCNJS.nextFontId++;
        window.WCNJS.fonts[id] = { name: nameStr, size: font_size };
        setValue(out_id, id, 'i32');
        
        // 同时通知 Worker
        const callId = window.WCNJS.callId++;
        window.WCNJS.fontWorker.postMessage({
            id: callId,
            cmd: 'loadFont',
            args: { name: nameStr, size: font_size }
        });
        
        return true;
    } catch (e) {
        console.error("[WCN] Load font failed:", e);
        return false;
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

// 异步生成位图 - 二进制协议
// 请求: [cmd:1, id:4, fontId:4, cp:4, size:4] = 17 bytes
EM_JS(int, js_generate_bitmap_async, (int font_id, uint32_t codepoint, float size), {
    const callbackId = window.WCNJS.callId++;
    
    window.WCNJS.pendingCalls[callbackId] = function(data) {
        if (!window.WCNJS.results) window.WCNJS.results = {};
        window.WCNJS.results[callbackId] = data; // ArrayBuffer
    };
    
    // 构建二进制请求
    const buf = new ArrayBuffer(17);
    const dv = new DataView(buf);
    dv.setUint8(0, 2); // cmd=genBitmap
    dv.setUint32(1, callbackId, true);
    dv.setUint32(5, font_id, true);
    dv.setUint32(9, codepoint, true);
    dv.setFloat32(13, size, true);
    
    window.WCNJS.fontWorker.postMessage(buf, [buf]);
    return callbackId;
});

// 批量异步生成位图 - 二进制协议
// 请求: [cmd:1, id:4, fontId:4, size:4, count:2, cps:4*count]
EM_JS(int, js_generate_bitmap_batch_async, (int font_id, const uint32_t* codepoints, int count, float size), {
    const callbackId = window.WCNJS.callId++;
    
    window.WCNJS.pendingCalls[callbackId] = function(data) {
        if (!window.WCNJS.results) window.WCNJS.results = {};
        window.WCNJS.results[callbackId] = data;
    };
    
    // 构建二进制请求
    const buf = new ArrayBuffer(15 + count * 4);
    const dv = new DataView(buf);
    dv.setUint8(0, 3); // cmd=genBatch
    dv.setUint32(1, callbackId, true);
    dv.setUint32(5, font_id, true);
    dv.setFloat32(9, size, true);
    dv.setUint16(13, count, true);
    
    for (let i = 0; i < count; i++) {
        dv.setUint32(15 + i * 4, Module.HEAPU32[(codepoints >> 2) + i], true);
    }
    
    window.WCNJS.fontWorker.postMessage(buf, [buf]);
    return callbackId;
});

// 获取批量结果数量 - 解析二进制响应
// 响应格式: [id:4, count:2, items...]
EM_JS(int, js_get_batch_result_count, (int callbackId), {
    const data = window.WCNJS.results && window.WCNJS.results[callbackId];
    if (!data || !(data instanceof ArrayBuffer)) return -1;
    const dv = new DataView(data);
    return dv.getUint16(4, true);
});

// 获取批量结果中的单个位图 - 解析二进制
// Item格式: [w:2, h:2, offX:4, offY:4, adv:4, isColor:1, pixelLen:4, pixels...]
EM_JS(bool, js_get_batch_result_item, (int callbackId, int index,
                                       unsigned char** out_ptr, int* out_w, int* out_h,
                                       float* out_off_x, float* out_off_y, float* out_adv,
                                       bool* out_is_color), {
    const data = window.WCNJS.results && window.WCNJS.results[callbackId];
    if (!data || !(data instanceof ArrayBuffer)) return false;
    
    const dv = new DataView(data);
    const count = dv.getUint16(4, true);
    if (index < 0 || index >= count) return false;
    
    // 遍历找到第 index 个 item
    let offset = 6;
    const itemHdr = 21; // w+h+offX+offY+adv+isColor+pixelLen
    for (let i = 0; i < index; i++) {
        const pixelLen = dv.getUint32(offset + 17, true);
        offset += itemHdr + pixelLen;
    }
    
    const w = dv.getUint16(offset, true);
    const h = dv.getUint16(offset + 2, true);
    if (w === 0 && h === 0) return false;
    
    const offX = dv.getFloat32(offset + 4, true);
    const offY = dv.getFloat32(offset + 8, true);
    const adv = dv.getFloat32(offset + 12, true);
    const isColor = dv.getUint8(offset + 16);
    const pixelLen = dv.getUint32(offset + 17, true);
    
    const ptr = Module._malloc(pixelLen);
    if (!ptr) return false;
    
    Module.HEAPU8.set(new Uint8Array(data, offset + 21, pixelLen), ptr);
    
    setValue(out_ptr, ptr, 'i32');
    setValue(out_w, w, 'i32');
    setValue(out_h, h, 'i32');
    setValue(out_off_x, offX, 'float');
    setValue(out_off_y, offY, 'float');
    setValue(out_adv, adv, 'float');
    if (out_is_color) setValue(out_is_color, isColor, 'i8');
    
    return true;
});

// 释放批量结果
EM_JS(void, js_free_batch_result, (int callbackId), {
    if (window.WCNJS.results) {
        delete window.WCNJS.results[callbackId];
    }
});

// 预加载常用字符 - 二进制协议
// 请求: [cmd:1, id:4, fontId:4, size:4] = 13 bytes
EM_JS(void, js_preload_common_glyphs, (int font_id, float size), {
    const callbackId = window.WCNJS.callId++;
    const buf = new ArrayBuffer(13);
    const dv = new DataView(buf);
    dv.setUint8(0, 4); // cmd=preload
    dv.setUint32(1, callbackId, true);
    dv.setUint32(5, font_id, true);
    dv.setFloat32(9, size, true);
    window.WCNJS.fontWorker.postMessage(buf, [buf]);
});

// 清除 Worker 缓存 - 二进制协议
EM_JS(void, js_clear_glyph_cache, (), {
    const callbackId = window.WCNJS.callId++;
    const buf = new ArrayBuffer(5);
    const dv = new DataView(buf);
    dv.setUint8(0, 5); // cmd=clearCache
    dv.setUint32(1, callbackId, true);
    window.WCNJS.fontWorker.postMessage(buf, [buf]);
});

// 获取异步位图结果 - 解析二进制响应
// 响应格式: [id:4, status:1, w:2, h:2, offX:4, offY:4, adv:4, isColor:1, pixels...]
EM_JS(bool, js_get_bitmap_result, (int callbackId,
                                   unsigned char** out_ptr, int* out_w, int* out_h,
                                   float* out_off_x, float* out_off_y, float* out_adv,
                                   bool* out_is_color), {
    const data = window.WCNJS.results && window.WCNJS.results[callbackId];
    if (!data) return false;
    
    delete window.WCNJS.results[callbackId];
    
    if (!(data instanceof ArrayBuffer)) return false;
    
    const dv = new DataView(data);
    const status = dv.getUint8(4);
    if (status === 0) return false;
    
    const w = dv.getUint16(5, true);
    const h = dv.getUint16(7, true);
    const offX = dv.getFloat32(9, true);
    const offY = dv.getFloat32(13, true);
    const adv = dv.getFloat32(17, true);
    const isColor = dv.getUint8(21);
    
    const pixelSize = w * h * 4;
    const ptr = Module._malloc(pixelSize);
    if (!ptr) return false;
    
    Module.HEAPU8.set(new Uint8Array(data, 22, pixelSize), ptr);
    
    setValue(out_ptr, ptr, 'i32');
    setValue(out_w, w, 'i32');
    setValue(out_h, h, 'i32');
    setValue(out_off_x, offX, 'float');
    setValue(out_off_y, offY, 'float');
    setValue(out_adv, adv, 'float');
    if (out_is_color) setValue(out_is_color, isColor, 'i8');
    
    return true;
});

// 主线程缓存辅助函数
EM_JS(void*, js_main_cache_get, (int font_id, uint32_t codepoint, float size), {
    const key = font_id + '_' + codepoint + '_' + Math.round(size);
    const cached = window.WCNJS.mainCache.get(key);
    if (cached) {
        // LRU: 移到末尾
        window.WCNJS.mainCache.delete(key);
        window.WCNJS.mainCache.set(key, cached);
        return cached.ptr;
    }
    return 0;
});

EM_JS(void, js_main_cache_set, (int font_id, uint32_t codepoint, float size, void* ptr, int w, int h, float offX, float offY, float adv, bool isColor), {
    const key = font_id + '_' + codepoint + '_' + Math.round(size);
    if (window.WCNJS.mainCache.size >= window.WCNJS.MAIN_CACHE_MAX) {
        const oldest = window.WCNJS.mainCache.keys().next().value;
        const oldEntry = window.WCNJS.mainCache.get(oldest);
        if (oldEntry && oldEntry.ptr) Module._free(oldEntry.ptr);
        window.WCNJS.mainCache.delete(oldest);
    }
    window.WCNJS.mainCache.set(key, { ptr, w, h, offX, offY, adv, isColor });
});

EM_JS(bool, js_main_cache_get_info, (int font_id, uint32_t codepoint, float size,
                                     int* out_w, int* out_h, float* out_off_x, float* out_off_y,
                                     float* out_adv, bool* out_is_color), {
    const key = font_id + '_' + codepoint + '_' + Math.round(size);
    const cached = window.WCNJS.mainCache.get(key);
    if (!cached) return false;
    setValue(out_w, cached.w, 'i32');
    setValue(out_h, cached.h, 'i32');
    setValue(out_off_x, cached.offX, 'float');
    setValue(out_off_y, cached.offY, 'float');
    setValue(out_adv, cached.adv, 'float');
    if (out_is_color) setValue(out_is_color, cached.isColor, 'i8');
    return true;
});

// 同步生成位图 (主线程回退，带缓存)
EM_JS(bool, js_generate_bitmap, (int font_id, uint32_t codepoint, float size,
                                unsigned char** out_ptr, int* out_w, int* out_h,
                                float* out_off_x, float* out_off_y, float* out_adv,
                                bool* out_is_color_ptr), {
    try {
        const font = window.WCNJS.fonts[font_id];
        if (!font) return false;
        
        // 检查缓存
        const cacheKey = font_id + '_' + codepoint + '_' + Math.round(size);
        const cached = window.WCNJS.mainCache.get(cacheKey);
        if (cached) {
            window.WCNJS.mainCache.delete(cacheKey);
            window.WCNJS.mainCache.set(cacheKey, cached);
            
            // 复制缓存数据
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

        // 存入缓存 (复制一份)
        if (window.WCNJS.mainCache.size >= window.WCNJS.MAIN_CACHE_MAX) {
            const oldest = window.WCNJS.mainCache.keys().next().value;
            const oldEntry = window.WCNJS.mainCache.get(oldest);
            if (oldEntry && oldEntry.ptr) Module._free(oldEntry.ptr);
            window.WCNJS.mainCache.delete(oldest);
        }
        const cachePtr = Module._malloc(bufSize);
        if (cachePtr) {
            Module.HEAPU8.copyWithin(cachePtr, ptr, ptr + bufSize);
            window.WCNJS.mainCache.set(cacheKey, { ptr: cachePtr, w, h, offX, offY, adv: metrics.width, isColor });
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

// ============================================================================
// 异步 API
// ============================================================================

#ifdef __EMSCRIPTEN__

// 异步版本 - 发起单个请求
int wcn_wasm_get_glyph_sdf_async(WCN_FontFace* face, uint32_t codepoint, float font_size) {
    if (!face) return -1;
    
    WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;
    return js_generate_bitmap_async(font_data->js_id, codepoint, font_size);
}

// 异步版本 - 检查结果
bool wcn_wasm_check_glyph_sdf_ready(int request_id) {
    return js_check_async_result(request_id);
}

// 异步版本 - 获取结果
bool wcn_wasm_get_glyph_sdf_result(int request_id,
                                   unsigned char** out_bitmap,
                                   int* out_width, int* out_height,
                                   float* out_offset_x, float* out_offset_y,
                                   float* out_advance,
                                   bool* out_is_color) {
    if (!out_bitmap || !out_width || !out_height) return false;
    
    if (out_is_color) *out_is_color = false;
    
    return js_get_bitmap_result(request_id, out_bitmap, out_width, out_height,
                               out_offset_x, out_offset_y, out_advance, out_is_color);
}

// ============================================================================
// 批量 API
// ============================================================================

// 批量异步 - 发起请求
int wcn_wasm_get_glyph_sdf_batch_async(WCN_FontFace* face, const uint32_t* codepoints, int count, float font_size) {
    if (!face || !codepoints || count <= 0) return -1;
    
    WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;
    return js_generate_bitmap_batch_async(font_data->js_id, codepoints, count, font_size);
}

// 批量异步 - 获取结果数量
int wcn_wasm_get_batch_result_count(int request_id) {
    return js_get_batch_result_count(request_id);
}

// 批量异步 - 获取单个结果
bool wcn_wasm_get_batch_result_item(int request_id, int index,
                                    unsigned char** out_bitmap,
                                    int* out_width, int* out_height,
                                    float* out_offset_x, float* out_offset_y,
                                    float* out_advance,
                                    bool* out_is_color) {
    if (!out_bitmap || !out_width || !out_height) return false;
    
    if (out_is_color) *out_is_color = false;
    
    return js_get_batch_result_item(request_id, index, out_bitmap, out_width, out_height,
                                   out_offset_x, out_offset_y, out_advance, out_is_color);
}

// 批量异步 - 释放结果
void wcn_wasm_free_batch_result(int request_id) {
    js_free_batch_result(request_id);
}

// ============================================================================
// 预加载 & 缓存管理
// ============================================================================

// 预加载常用字符 (ASCII + 常用中文标点)
void wcn_wasm_preload_common_glyphs(WCN_FontFace* face, float font_size) {
    if (!face) return;
    
    WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;
    js_preload_common_glyphs(font_data->js_id, font_size);
}

// 清除字形缓存
void wcn_wasm_clear_glyph_cache(void) {
    js_clear_glyph_cache();
}

#endif

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
#endif
