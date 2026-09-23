// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! KTX2 files with dapple's own mip chains.
//!
//! [`write()`] produces a KTX 2.0 container (Khronos KTX File Format
//! Specification 2.0) for one [`EncodedTexture`]:
//!
//! - a single 2D texture: `pixelDepth` 0, `layerCount` 0, `faceCount` 1;
//! - every level of the texture's chain, uncompressed (no
//!   supercompression);
//! - the Vulkan format of its [`PixelFormat`]: `VK_FORMAT_R8_UNORM`,
//!   `R8G8_UNORM`, `R8G8B8A8_UNORM`, `R8G8B8A8_SRGB`, `R16_UNORM` or
//!   `R32_SFLOAT`;
//! - a Basic Data Format Descriptor (KDF 1.3) with one sample per channel,
//!   BT.709 primaries, and the sRGB or linear transfer function; in sRGB
//!   files the alpha sample is flagged linear;
//! - a `KTXwriter` key naming this crate, so the file identifies its origin;
//! - levels stored smallest first, each at an offset aligned to
//!   `lcm(texel size, 4)`, as the specification requires, and indexed
//!   largest first.
//!
//! The 8-bit formats are the uncompressed formats `lightweald_texture_io`
//! reads; the 16-bit and float formats carry data such as height.
//! Block-compressed formats are a later step.

use alloc::vec::Vec;

use crate::pack::EncodedTexture;
use crate::quantize::PixelFormat;

/// The 12-byte KTX 2.0 file identifier.
pub const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// The `KTXwriter` value written into every file.
pub const WRITER: &str = "dapple_encode";

/// The Vulkan format number (`VkFormat`) of `format`.
#[must_use]
pub const fn vk_format(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::R8Unorm => 9,
        PixelFormat::Rg8Unorm => 16,
        PixelFormat::Rgba8Unorm => 37,
        PixelFormat::Rgba8Srgb => 43,
        PixelFormat::R16Unorm => 70,
        PixelFormat::R32Float => 100,
    }
}

const HEADER_BYTES: usize = 12 + 9 * 4 + 4 * 4 + 2 * 8;
const LEVEL_INDEX_ENTRY: usize = 3 * 8;

fn align(offset: usize, to: usize) -> usize {
    offset.div_ceil(to) * to
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn len_u32(n: usize) -> u32 {
    u32::try_from(n).expect("KTX2 section fits u32")
}

/// The Basic Data Format Descriptor block, prefixed by its total size.
fn data_format_descriptor(format: PixelFormat) -> Vec<u8> {
    let channels = format.channels();
    let bits = len_u32(format.bytes_per_channel() * 8);
    let float = format == PixelFormat::R32Float;
    let block_bytes = 24 + 16 * channels;
    let mut out = Vec::with_capacity(4 + block_bytes);
    put_u32(&mut out, len_u32(4 + block_bytes));
    // vendorId 0 (Khronos), descriptorType 0 (basic).
    put_u32(&mut out, 0);
    // versionNumber 2 (KDF 1.3), descriptorBlockSize.
    put_u32(&mut out, 2 | (len_u32(block_bytes) << 16));
    // colorModel RGBSDA (1), primaries BT.709 (1), transfer, flags 0
    // (straight alpha).
    let transfer: u32 = if format.is_srgb() { 2 } else { 1 };
    put_u32(&mut out, 1 | (1 << 8) | (transfer << 16));
    // texelBlockDimension0..3, each stored minus one: a 1×1×1×1 block.
    put_u32(&mut out, 0);
    // bytesPlane0 is the texel size; planes 1..7 are unused.
    put_u32(&mut out, len_u32(format.bytes_per_texel()));
    put_u32(&mut out, 0);
    for channel in 0..channels {
        let id: u32 = if channel == 3 { 15 } else { len_u32(channel) };
        // The alpha of an sRGB format is linear.
        let linear: u32 = if format.is_srgb() && channel == 3 {
            0x10
        } else {
            0
        };
        // Floats are signed; their range is written as float bits.
        let qualifiers: u32 = if float { 0xC0 } else { 0 };
        let bit_offset = len_u32(channel) * bits;
        // bitOffset, bitLength − 1, channelType with qualifiers.
        put_u32(
            &mut out,
            bit_offset | ((bits - 1) << 16) | ((id | linear | qualifiers) << 24),
        );
        // samplePosition0..3.
        put_u32(&mut out, 0);
        // sampleLower, sampleUpper.
        if float {
            put_u32(&mut out, (-1.0_f32).to_bits());
            put_u32(&mut out, 1.0_f32.to_bits());
        } else {
            put_u32(&mut out, 0);
            put_u32(&mut out, (1_u32 << bits) - 1);
        }
    }
    out
}

/// Key/value data: the `KTXwriter` entry, padded to four bytes.
fn key_value_data() -> Vec<u8> {
    let mut entry = Vec::new();
    entry.extend_from_slice(b"KTXwriter\0");
    entry.extend_from_slice(WRITER.as_bytes());
    entry.push(0);
    let mut out = Vec::new();
    put_u32(&mut out, len_u32(entry.len()));
    out.extend_from_slice(&entry);
    out.resize(align(out.len(), 4), 0);
    out
}

/// Writes `texture` and its mip chain as a KTX2 file.
///
/// # Panics
///
/// Panics when the texture has no levels or a level's length does not match
/// its size and format, which [`crate::pack()`] never produces.
#[must_use]
pub fn write(texture: &EncodedTexture) -> Vec<u8> {
    let levels = texture.levels.len();
    assert!(levels > 0, "a texture has at least one level");
    let texel = texture.format.bytes_per_texel();
    for (index, level) in texture.levels.iter().enumerate() {
        let w = (texture.width >> index).max(1) as usize;
        let h = (texture.height >> index).max(1) as usize;
        assert_eq!(level.len(), w * h * texel, "level {index} size");
    }
    let dfd = data_format_descriptor(texture.format);
    let kvd = key_value_data();

    let dfd_offset = HEADER_BYTES + levels * LEVEL_INDEX_ENTRY;
    let kvd_offset = dfd_offset + dfd.len();
    let level_alignment = match texel {
        1 | 2 | 4 => 4,
        _ => unreachable!("texel sizes are 1, 2 or 4 bytes"),
    };
    // Levels are stored smallest first.
    let mut offsets = alloc::vec![0_usize; levels];
    let mut cursor = kvd_offset + kvd.len();
    for index in (0..levels).rev() {
        cursor = align(cursor, level_alignment);
        offsets[index] = cursor;
        cursor += texture.levels[index].len();
    }

    let mut out = Vec::with_capacity(cursor);
    out.extend_from_slice(&IDENTIFIER);
    put_u32(&mut out, vk_format(texture.format));
    // typeSize: the size of one component, for endianness conversion.
    put_u32(&mut out, len_u32(texture.format.bytes_per_channel()));
    put_u32(&mut out, texture.width);
    put_u32(&mut out, texture.height);
    put_u32(&mut out, 0); // pixelDepth
    put_u32(&mut out, 0); // layerCount
    put_u32(&mut out, 1); // faceCount
    put_u32(&mut out, len_u32(levels));
    put_u32(&mut out, 0); // supercompressionScheme: none
    put_u32(&mut out, len_u32(dfd_offset));
    put_u32(&mut out, len_u32(dfd.len()));
    put_u32(&mut out, len_u32(kvd_offset));
    put_u32(&mut out, len_u32(kvd.len()));
    put_u64(&mut out, 0); // sgdByteOffset
    put_u64(&mut out, 0); // sgdByteLength
    for (index, level) in texture.levels.iter().enumerate() {
        put_u64(&mut out, offsets[index] as u64);
        put_u64(&mut out, level.len() as u64);
        put_u64(&mut out, level.len() as u64);
    }
    debug_assert_eq!(out.len(), dfd_offset, "level index ends at the DFD");
    out.extend_from_slice(&dfd);
    out.extend_from_slice(&kvd);
    for index in (0..levels).rev() {
        out.resize(offsets[index], 0);
        out.extend_from_slice(&texture.levels[index]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use dapple_raster::Edge;

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    fn u64_at(b: &[u8], at: usize) -> usize {
        usize::try_from(u64::from_le_bytes(b[at..at + 8].try_into().unwrap())).unwrap()
    }

    fn texture(format: PixelFormat, width: u32, height: u32) -> EncodedTexture {
        let texel = format.bytes_per_texel();
        let mut levels = Vec::new();
        let (mut w, mut h) = (width, height);
        let mut seed = 0_u8;
        loop {
            let n = w as usize * h as usize * texel;
            levels.push(
                (0..n)
                    .map(|i| seed.wrapping_add(u8::try_from(i % 256).unwrap()))
                    .collect(),
            );
            seed = seed.wrapping_add(101);
            if w == 1 && h == 1 {
                break;
            }
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        EncodedTexture {
            name: "test",
            format,
            width,
            height,
            edge: Edge::Wrap,
            levels,
        }
    }

    #[test]
    fn header_index_and_levels_read_back() {
        for format in [
            PixelFormat::R8Unorm,
            PixelFormat::Rg8Unorm,
            PixelFormat::Rgba8Unorm,
            PixelFormat::Rgba8Srgb,
            PixelFormat::R16Unorm,
            PixelFormat::R32Float,
        ] {
            let tex = texture(format, 8, 3);
            let file = write(&tex);
            assert_eq!(&file[..12], &IDENTIFIER);
            assert_eq!(u32_at(&file, 12), vk_format(format));
            assert_eq!(u32_at(&file, 16) as usize, format.bytes_per_channel());
            assert_eq!((u32_at(&file, 20), u32_at(&file, 24)), (8, 3));
            assert_eq!((u32_at(&file, 28), u32_at(&file, 32)), (0, 0));
            assert_eq!(u32_at(&file, 36), 1);
            assert_eq!(u32_at(&file, 40), 4, "8×3, 4×1, 2×1, 1×1");
            assert_eq!(u32_at(&file, 44), 0);
            let (dfd_off, dfd_len) = (u32_at(&file, 48) as usize, u32_at(&file, 52) as usize);
            let (kvd_off, kvd_len) = (u32_at(&file, 56) as usize, u32_at(&file, 60) as usize);
            assert_eq!(dfd_off, 80 + 4 * 24);
            assert_eq!(u32_at(&file, dfd_off) as usize, dfd_len);
            assert_eq!(kvd_off, dfd_off + dfd_len);
            assert!(
                file[kvd_off..kvd_off + kvd_len]
                    .windows(9)
                    .any(|w| w == b"KTXwriter")
            );

            // Levels: indexed largest first, stored smallest first, aligned.
            let mut previous_offset = usize::MAX;
            for (index, level) in tex.levels.iter().enumerate() {
                let entry = 80 + index * 24;
                let (offset, length) = (u64_at(&file, entry), u64_at(&file, entry + 8));
                assert_eq!(u64_at(&file, entry + 16), length);
                assert_eq!(offset % 4, 0);
                assert!(offset < previous_offset, "smaller levels come first");
                previous_offset = offset;
                assert_eq!(&file[offset..offset + length], level.as_slice());
            }
            assert!(previous_offset >= kvd_off + kvd_len);
        }
    }

    #[test]
    fn descriptor_declares_channels_and_transfer() {
        let dfd = data_format_descriptor(PixelFormat::Rgba8Srgb);
        assert_eq!(dfd.len(), 4 + 24 + 4 * 16);
        assert_eq!(u32_at(&dfd, 8) >> 16, 24 + 64, "descriptorBlockSize");
        assert_eq!((u32_at(&dfd, 12) >> 16) & 0xFF, 2, "sRGB transfer");
        assert_eq!(u32_at(&dfd, 20), 4, "bytesPlane0");
        let channel_type = |sample: usize| u32_at(&dfd, 28 + sample * 16) >> 24;
        assert_eq!(
            [0, 1, 2, 3].map(channel_type),
            [0, 1, 2, 15 | 0x10],
            "RGB, then alpha flagged linear"
        );
        let float = data_format_descriptor(PixelFormat::R32Float);
        assert_eq!(u32_at(&float, 28) >> 24, 0xC0, "signed float red");
        assert_eq!((u32_at(&float, 28) >> 16) & 0xFF, 31, "32 bits");
        assert_eq!(u32_at(&float, 40), 1.0_f32.to_bits(), "sampleUpper");
        let short = data_format_descriptor(PixelFormat::R16Unorm);
        assert_eq!(u32_at(&short, 40), 65535, "sampleUpper");
        let linear = data_format_descriptor(PixelFormat::Rg8Unorm);
        assert_eq!((u32_at(&linear, 12) >> 16) & 0xFF, 1, "linear transfer");
        assert_eq!(u32_at(&linear, 28 + 16) & 0xFFFF, 8, "G starts at bit 8");
    }

    #[test]
    fn single_level_textures_write() {
        let tex = EncodedTexture {
            name: "one",
            format: PixelFormat::R8Unorm,
            width: 1,
            height: 1,
            edge: Edge::Clamp,
            levels: vec![vec![42]],
        };
        let file = write(&tex);
        assert_eq!(u32_at(&file, 40), 1);
        assert_eq!(*file.last().unwrap(), 42);
    }
}
