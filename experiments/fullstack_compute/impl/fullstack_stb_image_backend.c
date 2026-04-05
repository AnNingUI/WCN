#include "fullstack_stb_image_backend.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

static bool fs_stb_decode_image_memory(
    const uint8_t* encoded_bytes,
    size_t encoded_size,
    uint8_t** out_rgba_pixels,
    uint32_t* out_width,
    uint32_t* out_height
) {
    if (!encoded_bytes || encoded_size == 0u || !out_rgba_pixels || !out_width || !out_height) {
        return false;
    }
    *out_rgba_pixels = NULL;
    *out_width = 0u;
    *out_height = 0u;

    int w = 0;
    int h = 0;
    int ch = 0;
    stbi_uc* decoded = stbi_load_from_memory((const stbi_uc*)encoded_bytes, (int)encoded_size, &w, &h, &ch, 4);
    if (!decoded || w <= 0 || h <= 0) {
        if (decoded) {
            stbi_image_free(decoded);
        }
        return false;
    }
    *out_rgba_pixels = (uint8_t*)decoded;
    *out_width = (uint32_t)w;
    *out_height = (uint32_t)h;
    return true;
}

static bool fs_stb_decode_image_file(
    const char* path,
    uint8_t** out_rgba_pixels,
    uint32_t* out_width,
    uint32_t* out_height
) {
    if (!path || !out_rgba_pixels || !out_width || !out_height) {
        return false;
    }
    *out_rgba_pixels = NULL;
    *out_width = 0u;
    *out_height = 0u;

    int w = 0;
    int h = 0;
    int ch = 0;
    stbi_uc* decoded = stbi_load(path, &w, &h, &ch, 4);
    if (!decoded || w <= 0 || h <= 0) {
        if (decoded) {
            stbi_image_free(decoded);
        }
        return false;
    }
    *out_rgba_pixels = (uint8_t*)decoded;
    *out_width = (uint32_t)w;
    *out_height = (uint32_t)h;
    return true;
}

static void fs_stb_free_image(uint8_t* rgba_pixels) {
    if (rgba_pixels) {
        stbi_image_free(rgba_pixels);
    }
}

static const FS_ImageBackend g_stb_image_backend = {
    .name = "stb_image",
    .decode_memory = fs_stb_decode_image_memory,
    .decode_file = fs_stb_decode_image_file,
    .free_image = fs_stb_free_image
};

const FS_ImageBackend* fs_get_stb_image_backend(void) {
    return &g_stb_image_backend;
}
