use std::{convert::TryFrom, mem::size_of, os::fd::AsRawFd};

use drm::control::framebuffer::Handle;
use drm::SystemError;
use drm_fourcc::{DrmFourcc, DrmModifier};
use image::{GenericImage, RgbImage};
use libc::close;
use nix::sys::mman;

use crate::{
    ffi::{self, gem_close},
    image_decoder::{
        decode_image, decode_image_multichannel, decode_small_image_multichannel,
        decode_tiled_small_image, rgb565_to_rgb888, ToRgb, YUV420Pixel,
    },
    Card,
};

fn copy_buffer<T: Sized + Copy>(
    card: &Card,
    handle: u32,
    to: &mut [T],
    verbose: bool,
) -> Result<(), SystemError> {
    let length = to.len() * size_of::<T>();

    let hfd = ffi::prime_handle_to_fd(card.as_raw_fd(), handle)?;

    if verbose {
        println!("handle fd {}", hfd);
    }

    let addr = core::ptr::null_mut();
    let prot = mman::ProtFlags::PROT_READ;
    let flags = mman::MapFlags::MAP_SHARED;
    unsafe {
        let map = mman::mmap(addr, length as _, prot, flags, hfd, 0).unwrap();

        let mapping: &mut [T] = std::slice::from_raw_parts_mut(map as *mut _, to.len());
        to.copy_from_slice(mapping);
        mman::munmap(map, length as _).unwrap();

        if close(hfd) == -1 {
            panic!("Failed to close prime fd.");
        };
    }

    Ok(())
}

fn decimate_image_4(size: (usize, usize), image: &[u32], copy: &mut [u32]) {
    let decim = (4, 4);
    let newsize = (size.0 / decim.0, size.1 / decim.1);

    for y in 0..newsize.1 {
        let ty = decim.1 * y;
        for x in 0..newsize.0 {
            let tx = decim.0 * x;
            copy[y * newsize.0 + x] = image[ty * size.0 + tx];
        }
    }
}


fn decode_xrgb2101010_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    handle: u32,
    modifier: u64, // Añadir el modificador como parámetro
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let bytes_per_pixel = 4; // Cada pixel es de 4 bytes (32 bits)
    let length = (pitch * size.1) / bytes_per_pixel; // En elementos u32

    if verbose {
        println!(
            "XRGB2101010, size: {:?}, pitch: {}, length: {}, modifier: {}",
            size, pitch, length, modifier
        );
    }

    let mut copy = vec![0u32; length as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    let decimate = 16; // Decimar la imagen por 4
    // let mut img = RgbImage::new(size.0, size.1);
    let mut dec = RgbImage::new(size.0 / decimate, size.1 / decimate);


    for y in 0..size.1 {
        for x in 0..size.0 {
            // Si x o y no son múltiplos de decimate, saltar el píxel
            // Decimar la imagen
            if x % decimate != 0 || y % decimate != 0 {
                continue;
            }

            let index = if modifier == 72057594037927938 { // I915_FORMAT_MOD_Y_TILED
                calculate_y_tiled_index(x, y, size.0, size.1, pitch)
            } else {
                (y * (pitch / bytes_per_pixel) + x) as usize
            };

            let pixel = copy.get(index).cloned().unwrap_or(0);

            // Extraer los componentes de color de 10 bits
            let r = ((pixel >> 20) & 0x3FF) as u32;
            let g = ((pixel >> 10) & 0x3FF) as u32;
            let b = (pixel & 0x3FF) as u32;

            // Convertir de 10 bits a 8 bits
            let r = (r * 255 / 1023) as u8;
            let g = (g * 255 / 1023) as u8;
            let b = (b * 255 / 1023) as u8;

            // Colocar el píxel en la imagen
            // img.put_pixel(x, y, image::Rgb([r, g, b]));

            // Decimar la imagen
            dec.put_pixel(x / decimate, y / decimate, image::Rgb([r, g, b]));
        }
    }

    Ok(dec)
}

fn calculate_y_tiled_indexx(x: u32, y: u32, width: u32, height: u32, pitch: u32) -> usize {
    // Definimos los parámetros del tile, los mismos que en tu código anterior
    const TILE_WIDTH: u32 = 32; // 32 píxeles de ancho por tile
    const TILE_HEIGHT: u32 = 16; // 16 píxeles de alto por tile
    const BYTES_PER_PIXEL: u32 = 4; // XRGB2101010 es de 4 bytes por píxel

    // Calculamos el índice del tile
    let tile_x = x / TILE_WIDTH;
    let tile_y = y / TILE_HEIGHT;

    // Calculamos la posición dentro del tile
    let within_tile_x = x % TILE_WIDTH;
    let within_tile_y = y % TILE_HEIGHT;

    // Calculamos cuántos tiles hay por fila de la imagen (ancho en tiles)
    let tiles_per_row = pitch / (TILE_WIDTH * BYTES_PER_PIXEL);

    // Calculamos el índice del tile en la imagen completa
    let tile_index = (tile_y * tiles_per_row) + tile_x;

    // Desplazamiento dentro del tile
    let tile_offset = tile_index * TILE_WIDTH * TILE_HEIGHT * BYTES_PER_PIXEL;
    let pixel_offset = (within_tile_y * TILE_WIDTH + within_tile_x) * BYTES_PER_PIXEL;

    // Retornamos el índice correcto en el buffer
    (tile_offset + pixel_offset) as usize
}


// Este tiene el tamaño correcto pero borroso
fn calculate_y_tiled_index(x: u32, y: u32, width: u32, height: u32, pitch: u32) -> usize {
    const TILE_WIDTH: u32 = 128;  // 128 bytes (32 píxeles para XRGB2101010)
    const TILE_HEIGHT: u32 = 32;  // 32 filas de píxeles por tile
    const TILE_PIXELS: u32 = TILE_WIDTH * TILE_HEIGHT / 4;  // 4096 bytes, o 1024 píxeles por tile

    let tile_x = (x / 32);  // Cada tile tiene 32 píxeles de ancho (32*4 bytes por píxel = 128 bytes por fila)
    let tile_y = (y / TILE_HEIGHT);

    let within_tile_x = x % 32;
    let within_tile_y = y % TILE_HEIGHT;

    let num_tiles_in_row = (pitch / TILE_WIDTH); // Número de tiles por fila completa

    let tile_index = tile_y * num_tiles_in_row + tile_x;

    let pixel_offset_within_tile = (within_tile_y * 32 + within_tile_x);

    // Índice final
    (tile_index * TILE_PIXELS + pixel_offset_within_tile) as usize
}


// Este es más preciso pero se ve doble la imagen
fn calculate_y_tiled_index1(x: u32, y: u32, width: u32, height: u32, pitch: u32) -> usize {
    const TILE_WIDTH: usize = 32;
    const TILE_HEIGHT: usize = 16;
    const BYTES_PER_PIXEL: usize = 4;
    const TILE_SIZE: usize = TILE_WIDTH * TILE_HEIGHT * BYTES_PER_PIXEL; // 4 bytes por píxel

    let tile_x = x / TILE_WIDTH as u32;
    let tile_y = y / TILE_HEIGHT as u32;

    let within_tile_x = x % TILE_WIDTH as u32;
    let within_tile_y = y % TILE_HEIGHT as u32;

    let tile_index = tile_y * (width / TILE_WIDTH as u32) + tile_x;

    let tile_offset = tile_index as usize * TILE_SIZE; // (Este parametro da la forma de la imagen)
    let pixel_offset = (within_tile_x * TILE_HEIGHT as u32 + within_tile_y) * 4; // (Este parametro da resolucion)

    (tile_offset + pixel_offset as usize) as usize
}



fn decode_p030_image(
    card: &Card,
    size: (usize, usize),
    pitches: u32,
    handle: u32,
    modifier: u64,
    offset: usize,
    verbose: bool,
) -> Result<RgbImage, SystemError> {

    // We assume the DRM BROADCOM SAND128 format
    if u64::from(drm_fourcc::DrmModifier::Broadcom_sand128) != modifier & !(0xFFFF << 8) {
        panic!("Unsupported P030 modifier value");
    }

    let stride = 128 / 4; // each column is 128 bytes wide, we use 4 bytes per word
    let colpx = 96;

    let ypitch = pitches as usize / (32 / 8);
    let ylines = ((modifier >> 8) & 0xFFFFFFFF) as usize;
    let length = ylines * (size.0 / colpx) * stride;
    let crcboffset = offset / 4; // offset of the CrCb information in each column

    if verbose {
        println!(
            "P030, size: {:?}, lines: {}, pitches: {}, length: {}",
            size, ylines, ypitch, length
        );
    }

    let mut yplane = vec![0u32; length as _];
    copy_buffer(card, handle, &mut yplane, verbose)?;

    let decim = 3;
    let mut img = RgbImage::new((size.0 / decim) as _, (size.1 / decim) as _);
    for y in 0..size.1 / decim {
        let ty = y * decim;
        for x in 0..size.0 / decim {
            let tx = x * decim;
            let col = tx / colpx;
            let col_offset = col * stride * ylines;
            let x_mod = (tx % colpx) / decim;

            let ypx = unsafe { yplane.get_unchecked(col_offset + ty * stride + x_mod) };
            let rx = x_mod / 2 * 2;
            let crcind = col_offset + crcboffset + ty / 2 * stride + rx;
            let crcbpx = unsafe { yplane.get_unchecked(crcind + 1) };

            let yuv = YUV420Pixel::new((ypx >> 2) as u8, (crcbpx >> 12) as u8, (crcbpx >> 2) as u8);

            unsafe {
                img.unsafe_put_pixel(x as _, y as _, yuv.rgb());
            }
        }
    }

    Ok(img)
}

fn decode_nv12_image(
    card: &Card,
    size: (usize, usize),
    pitches: u32,
    handle: u32,
    modifier: u64,
    offset: usize,
    verbose: bool,
) -> Result<RgbImage, SystemError> {

    // We assume the DRM BROADCOM SAND128 format
    if u64::from(drm_fourcc::DrmModifier::Broadcom_sand128) != modifier & !(0xFFFF << 8) {
        panic!("Unsupported NV12 modifier value");
    }

    let stride = 128 / 4; // each column is 128 bytes wide, we use 4 bytes per word
    let colpx = 128; // 1 byte per pixel

    let ypitch = pitches as usize / (32 / 8);
    let ylines = ((modifier >> 8) & 0xFFFFFFFF) as usize;
    let length = ylines * (size.0 / colpx) * stride;
    let crcboffset = offset / 4; // offset of the CrCb information in each column

    if verbose {
        println!(
            "NV12, size: {:?}, lines: {}, pitches: {}, length: {}",
            size, ylines, ypitch, length
        );
    }

    let mut yplane = vec![0u32; length as _];
    copy_buffer(card, handle, &mut yplane, verbose)?;

    let decim : usize = 4;
    let mut img = RgbImage::new((size.0 / decim) as _, (size.1 / decim) as _);
    for y in 0..size.1 / decim {
        let ty = y * decim;
        for x in 0..size.0 / decim {
            let tx = x * decim;
            let col = tx / colpx;
            let col_offset = col * stride * ylines;
            let x_mod = (tx % colpx) / decim;

            let ypx = unsafe { yplane.get_unchecked(col_offset + ty * stride + x_mod) };
            let rx = x_mod / 2 * 2;
            let crcind = col_offset + crcboffset + ty / 2 * stride + rx;
            let crcbpx = unsafe { yplane.get_unchecked(crcind + 1) };

            let yuv = YUV420Pixel::new((ypx >> 0) as u8, (crcbpx >> 0) as u8, (crcbpx >> 8) as u8);

            unsafe {
                img.unsafe_put_pixel(x as _, y as _, yuv.rgb());
            }
        }
    }

    Ok(img)
}


fn dump_linear_to_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    bpp: u32,
    handle: u32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let size = (size.0, size.1);

    let length = pitch * size.1 / (bpp / 8);

    println!(
        "linear, size: {:?}, pitch: {}, bpp: {}, length: {}",
        size, pitch, bpp, length
    );
    let mut copy = vec![0u32; length as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    let mut dec = vec![0u32; (length / (4 * 4)) as _];
    decimate_image_4(
        (size.0 as _, size.1 as _),
        copy.as_slice(),
        dec.as_mut_slice(),
    );

    Ok(decode_image(
        dec.as_mut_slice(),
        pitch / 4,
        (size.0 / 4, size.1 / 4),
    ))
}

fn dump_rgb565_to_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    bpp: u32,
    handle: u32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // let size = (size.0, size.1 / 64);

    let length = pitch * size.1 / (bpp / 8);

    println!(
        "rgb565, size: {:?}, pitch: {}, bpp: {}, length: {}",
        size, pitch, bpp, length
    );
    let mut copy = vec![0u16; length as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    Ok(rgb565_to_rgb888(copy.as_mut_slice(), pitch, size))
}

fn dump_broadcom_tiled_to_image(
    card: &Card,
    size: (u32, u32),
    bpp: u32,
    handle: u32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let tilesize = 32;
    let tile_count = |n| (n + tilesize - 1) / tilesize;
    let tiles = (tile_count(size.0), tile_count(size.1));
    let total_tiles = tiles.0 * tiles.1;

    let length = total_tiles * tilesize * tilesize * (bpp / 8);

    let mut copy = vec![0; (length / 4) as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    Ok(decode_tiled_small_image(
        copy.as_mut_slice(),
        tilesize,
        tiles,
        size,
    ))
}

fn dump_yuv420_to_image(
    card: &Card,
    size: (u32, u32),
    pitches: [u32; 4],
    handles: [u32; 4],
    offsets: [u32; 4],
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // The length of the entire buffer is the length of the last buffer plus its
    // offset (assuming they are in order). The U and V buffers are grouped into
    // 2x2 tiles, hence the length is divided by 4.
    let length = offsets[2] + size.1 * pitches[2] * pitches[2] / pitches[0];
    //println!("  -> Mounting @{} +{}", offset, length);

    let mut copy = vec![0; length as _];
    copy_buffer(card, handles[0], &mut copy, verbose)?;

    let buffer_range = |i| {
        offsets[i] as usize..(offsets[i] + size.1 * pitches[i] * pitches[i] / pitches[0]) as usize
    };

    let mappings = [
        &copy[buffer_range(0)],
        &copy[buffer_range(1)],
        &copy[buffer_range(2)],
    ];

    let mut pitches1 = [0; 3];
    pitches1.copy_from_slice(&pitches[0..3]);

    if size.0 > 640 {
        // If the image is large then just decode a smaller image
        Ok(decode_small_image_multichannel(mappings, size, pitches1))
    } else {
        Ok(decode_image_multichannel(mappings, size, pitches1))
    }
}

pub fn dump_framebuffer_to_image(
    card: &Card,
    fb: Handle,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let fbinfo2 = ffi::fb_cmd2(card.as_raw_fd(), fb.into())?;

    if verbose {
        println!("  -> FB Info 2: {:?}", fbinfo2);
    }

    let size = (fbinfo2.width, fbinfo2.height);

    if fbinfo2.pixel_format == 808661072 {
        return decode_p030_image(
            card,
            (size.0 as _, size.1 as _),
            fbinfo2.pitches[0],
            fbinfo2.handles[0],
            fbinfo2.modifier[0],
            fbinfo2.offsets[1] as _,
            verbose,
        );
    }

    let fourcc = drm_fourcc::DrmFourcc::try_from(fbinfo2.pixel_format).unwrap();
    let modifier = drm_fourcc::DrmModifier::try_from(fbinfo2.modifier[0]).unwrap();

    let image_result = match fourcc {
        DrmFourcc::Xrgb8888 => match modifier {
            DrmModifier::Broadcom_vc4_t_tiled => {
                dump_broadcom_tiled_to_image(card, size, 32, fbinfo2.handles[0], verbose)
            }
            DrmModifier::Linear => dump_linear_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                verbose,
            ),
            _ => panic!("Unsupported framebuffer modifier: {:?}", modifier),
        },
        DrmFourcc::Argb8888 => match modifier {
            DrmModifier::Broadcom_vc4_t_tiled => {
                dump_broadcom_tiled_to_image(card, size, 32, fbinfo2.handles[0], verbose)
            }
            DrmModifier::Linear => dump_linear_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                verbose,
            ),
            _ => panic!("Unsupported framebuffer modifier: {:?}", modifier),
        },
        DrmFourcc::Yuv420 => dump_yuv420_to_image(
            card,
            size,
            fbinfo2.pitches,
            fbinfo2.handles,
            fbinfo2.offsets,
            verbose,
        ),
        DrmFourcc::Rgb565 => dump_rgb565_to_image(
            card,
            fbinfo2.pitches[0],
            size,
            16,
            fbinfo2.handles[0],
            verbose,
        ),
        DrmFourcc::Nv12 => decode_nv12_image(
            card,
            (size.0 as _, size.1 as _),
            fbinfo2.pitches[0],
            fbinfo2.handles[0],
            fbinfo2.modifier[0],
            fbinfo2.offsets[1] as _,
            verbose
        ),
        DrmFourcc::Xrgb2101010 => decode_xrgb2101010_image(
            card,
            fbinfo2.pitches[0],
            size,
            fbinfo2.handles[0],
            fbinfo2.modifier[0],
            verbose
        ),
        _ => panic!(
            "Unsupported framebuffer pixel format: {} {:x}",
            fourcc, fbinfo2.pixel_format
        ),
    };

    gem_close(card.as_raw_fd(), fbinfo2.handles[0]).unwrap();

    let image = image_result?;

    Ok(image)
}
