#ifndef WCN_STB_IMAGE_IMPL_H
#define WCN_STB_IMAGE_IMPL_H

#include "WCN/WCN.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

#include <stdlib.h>
#include <string.h>

static bool wcn_stb_decode_image(
    const uint8_t* image_data,
    size_t data_size,
    WCN_ImageData* out_image
) {
    if (!image_data || data_size == 0 || !out_image) {
        return false;
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    stbi_uc* decoded = stbi_load_from_memory(
        image_data,
        (int)data_size,
        &width,
        &height,
        &channels,
        4
    );

    if (!decoded || width <= 0 || height <= 0) {
        if (decoded) {
            stbi_image_free(decoded);
        }
        return false;
    }

    const size_t rgba_size = (size_t)width * (size_t)height * 4;
    uint8_t* buffer = (uint8_t*)malloc(rgba_size);
    if (!buffer) {
        stbi_image_free(decoded);
        return false;
    }

    memcpy(buffer, decoded, rgba_size);
    stbi_image_free(decoded);

    out_image->data = buffer;
    out_image->width = (uint32_t)width;
    out_image->height = (uint32_t)height;
    out_image->format = 0; // RGBA8
    out_image->data_size = rgba_size;

    return true;
}

static WCN_ImageDecoder wcn_stb_image_decoder = {
    .decode = wcn_stb_decode_image,
    .name = "stb_image"
};

static inline WCN_ImageDecoder* wcn_get_stb_image_decoder(void) {
    return &wcn_stb_image_decoder;
}

#endif // WCN_STB_IMAGE_IMPL_H

