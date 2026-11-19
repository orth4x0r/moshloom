//! Asynchronous readback: a small picture of an image, without ever waiting for the GPU.
//!
//! [`Gpu::read_back`](crate::Gpu::read_back) allocates a buffer, submits a copy, waits for it and
//! converts half floats one channel at a time out of mapped memory: about 5 ms however small the
//! image (22 ms at 256x144 on the reference machine). That is fine for a test and wrong for
//! anything the render thread does while a show runs.
//!
//! An [`AsyncReadback`] does the same job inside a frame that is being recorded anyway:
//!
//! 1. [`AsyncReadback::record`] adds a few commands to the frame's own command buffer: the source
//!    is blitted down (linear filter, two steps when it is much larger than the picture) into a
//!    small 8-bit sRGB image, and that is copied into one of a few host-visible buffers that are
//!    allocated once and stay mapped. No submission of its own, no fence, no wait.
//! 2. [`AsyncReadback::begin_frame`] is called once per frame, before anything is recorded. It
//!    hands back the picture that was recorded [`READBACK_DELAY`] frames ago: by then the frame
//!    that carried the copy has long finished (the stage waits for every frame's layers before it
//!    presents, and submissions on one queue finish in order), so the bytes are simply there.
//!
//! The blit converts the format, so a linear RGBA16F layer target and the sRGB swapchain image
//! both come out as display-ready RGBA8.

use ash::vk;

use crate::device::{create_buffer, Allocations};
use crate::target::{cmd_barrier, Barrier};
use crate::{Gpu, RenderError};

/// A picture recorded in frame `n` is handed out by `begin_frame` in frame `n + READBACK_DELAY`.
pub const READBACK_DELAY: u64 = 2;

/// One buffer per frame a picture can be waiting in, plus the one being written.
const SLOTS: usize = READBACK_DELAY as usize + 1;

/// The picture's format: 8-bit sRGB, so the bytes are what a UI texture wants.
const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// A source this much larger than the picture (per axis) is blitted in two steps: one bilinear
/// step from far away skips most of the source's pixels and sparkles.
const TWO_STEP_FROM: u32 = 3;
/// The intermediate image is this many times the picture, per axis.
const MID_SCALE: u32 = 4;

/// One finished picture: tightly packed RGBA8, top row first, sRGB-encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Readback {
    /// What the caller passed to `record`, to tell pictures apart.
    pub tag: u64,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The image a readback is taken from, and the layouts it is in before and after.
#[derive(Clone, Copy)]
pub(crate) struct ReadbackSource {
    pub(crate) image: vk::Image,
    pub(crate) extent: vk::Extent2D,
    /// From the layout the image is in now to `TRANSFER_SRC_OPTIMAL`.
    pub(crate) before: Barrier,
    /// From `TRANSFER_SRC_OPTIMAL` to the layout the caller wants it left in.
    pub(crate) after: Barrier,
    /// Flip vertically on the way into the picture. A `LayerTarget`'s row 0 is the stage's
    /// bottom row (`Compositor` flips it into the swapchain), so thumbnails taken from a layer
    /// target flip here to match the stage; the swapchain preview is already upright.
    pub(crate) flip_y: bool,
}

struct Slot {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *const u8,
    /// The tag and the frame a picture was recorded in, until it is handed out.
    pending: Option<(u64, u64)>,
}

struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    extent: vk::Extent2D,
}

/// A small picture of some image, a couple of frames late and for almost nothing.
///
/// Must be dropped before the [`Gpu`] it was built on, and not while a command buffer that
/// `record` wrote to is still executing (the stage waits for the device first).
pub struct AsyncReadback {
    device: ash::Device,
    allocations: Allocations,
    small: Image,
    mid: Image,
    slots: [Slot; SLOTS],
    cursor: usize,
    frame: u64,
    recorded: u64,
    delivered: u64,
}

impl AsyncReadback {
    /// A readback that produces `width` x `height` pictures.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<AsyncReadback, RenderError> {
        let (width, height) = (width.max(1), height.max(1));
        let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let make = |extent: vk::Extent2D| -> Result<Image, RenderError> {
            let (image, memory) = gpu.create_image(extent, FORMAT, usage)?;
            Ok(Image {
                image,
                memory,
                extent,
            })
        };
        let small = make(vk::Extent2D { width, height })?;
        let mid = make(vk::Extent2D {
            width: width * MID_SCALE,
            height: height * MID_SCALE,
        })?;

        let bytes = u64::from(width) * u64::from(height) * 4;
        let allocations = gpu.allocations();
        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            // Cached memory reads at memcpy speed; a device that has none still gets a buffer.
            let visible =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let (buffer, memory) = create_buffer(
                &gpu.dev,
                gpu.memory_properties(),
                &allocations,
                bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                visible | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .or_else(|_| {
                create_buffer(
                    &gpu.dev,
                    gpu.memory_properties(),
                    &allocations,
                    bytes,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    visible,
                )
            })?;
            let mapped = unsafe {
                gpu.dev
                    .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())?
            } as *const u8;
            slots.push(Slot {
                buffer,
                memory,
                mapped,
                pending: None,
            });
        }
        let slots: [Slot; SLOTS] = slots
            .try_into()
            .map_err(|_| RenderError::Stage("readback slots".into()))?;
        Ok(AsyncReadback {
            device: gpu.dev.clone(),
            allocations,
            small,
            mid,
            slots,
            cursor: 0,
            frame: 0,
            recorded: 0,
            delivered: 0,
        })
    }

    /// The picture's size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.small.extent.width, self.small.extent.height)
    }

    /// Pictures recorded, and pictures handed out, since this readback was built.
    pub fn counts(&self) -> (u64, u64) {
        (self.recorded, self.delivered)
    }

    /// Start a frame: returns the picture recorded [`READBACK_DELAY`] frames ago, if there is
    /// one. Call it once per frame, before `record`. It never waits: it copies bytes the GPU
    /// finished writing frames ago.
    pub fn begin_frame(&mut self) -> Option<Readback> {
        self.frame += 1;
        let frame = self.frame;
        let (width, height) = self.size();
        let bytes = (width * height * 4) as usize;
        // The oldest picture that is old enough.
        let slot = self
            .slots
            .iter_mut()
            .filter(|slot| {
                slot.pending
                    .is_some_and(|(_, recorded)| recorded + READBACK_DELAY <= frame)
            })
            .min_by_key(|slot| slot.pending.map(|(_, recorded)| recorded))?;
        let (tag, _) = slot.pending.take()?;
        let mut rgba = vec![0u8; bytes];
        // The buffer is `bytes` long, mapped for the readback's lifetime, and the copy that
        // wrote it finished at least a frame ago.
        unsafe {
            std::ptr::copy_nonoverlapping(slot.mapped, rgba.as_mut_ptr(), bytes);
        }
        // The blit writes the source's alpha; a picture for a UI is opaque.
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }
        self.delivered += 1;
        Some(Readback {
            tag,
            width,
            height,
            rgba,
        })
    }

    /// Add the readback of `source` to a command buffer that is being recorded. The picture
    /// comes out of `begin_frame` [`READBACK_DELAY`] frames from now, provided the command
    /// buffer is submitted this frame.
    pub(crate) fn record(&mut self, cmd: vk::CommandBuffer, source: ReadbackSource, tag: u64) {
        let slot = self.cursor;
        self.cursor = (self.cursor + 1) % SLOTS;
        let device = &self.device;

        cmd_barrier(device, cmd, source.image, source.before);
        let two_step = source.extent.width > self.small.extent.width * TWO_STEP_FROM
            || source.extent.height > self.small.extent.height * TWO_STEP_FROM;
        let from = if two_step {
            blit(
                device,
                cmd,
                source.image,
                source.extent,
                self.mid.image,
                self.mid.extent,
                false,
            );
            (self.mid.image, self.mid.extent)
        } else {
            (source.image, source.extent)
        };
        if two_step {
            cmd_barrier(device, cmd, self.mid.image, written_to_source());
        }
        blit(
            device,
            cmd,
            from.0,
            from.1,
            self.small.image,
            self.small.extent,
            source.flip_y,
        );
        cmd_barrier(device, cmd, self.small.image, written_to_source());
        let region = vk::BufferImageCopy::default()
            .image_subresource(layers())
            .image_extent(vk::Extent3D {
                width: self.small.extent.width,
                height: self.small.extent.height,
                depth: 1,
            });
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                self.small.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.slots[slot].buffer,
                &[region],
            );
        }
        cmd_barrier(device, cmd, source.image, source.after);

        self.slots[slot].pending = Some((tag, self.frame));
        self.recorded += 1;
    }

    /// Record a readback of `target`'s current picture in a submission of its own, and wait for
    /// it. This is the test path (and the only way to use a readback without a stage): the
    /// commands are the ones `record` adds to a frame, so a test of this is a test of that.
    /// The picture still comes out of `begin_frame` [`READBACK_DELAY`] frames later.
    ///
    /// Returns how long recording the commands took: that, and not the wait this method adds,
    /// is what a frame pays for a picture.
    pub fn record_target(
        &mut self,
        gpu: &Gpu,
        target: &crate::LayerTarget,
        tag: u64,
    ) -> Result<std::time::Duration, RenderError> {
        let source = target_source(target);
        let mut recording = std::time::Duration::ZERO;
        gpu.one_shot(|cmd| {
            let started = std::time::Instant::now();
            self.record(cmd, source, tag);
            recording = started.elapsed();
        })?;
        target.set_front_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        Ok(recording)
    }
}

impl Drop for AsyncReadback {
    fn drop(&mut self) {
        unsafe {
            for slot in &self.slots {
                self.device.unmap_memory(slot.memory);
                self.device.destroy_buffer(slot.buffer, None);
                self.device.free_memory(slot.memory, None);
                self.allocations.release(slot.memory);
            }
            for image in [&self.small, &self.mid] {
                self.device.destroy_image(image.image, None);
                self.device.free_memory(image.memory, None);
                self.allocations.release(image.memory);
            }
        }
    }
}

/// A layer target's current picture as a readback source: whatever layout it is in now, and
/// back to the one every pass expects to sample it in.
pub(crate) fn target_source(target: &crate::LayerTarget) -> ReadbackSource {
    ReadbackSource {
        image: target.front_image(),
        extent: vk::Extent2D {
            width: target.width(),
            height: target.height(),
        },
        before: Barrier {
            old_layout: target.front_layout(),
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage: vk::PipelineStageFlags::TRANSFER,
            dst_access: vk::AccessFlags::TRANSFER_READ,
        },
        after: Barrier {
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_stage: vk::PipelineStageFlags::TRANSFER,
            src_access: vk::AccessFlags::TRANSFER_READ,
            dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            dst_access: vk::AccessFlags::SHADER_READ,
        },
        flip_y: true,
    }
}

fn layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
}

/// An image this module owns, just written by a blit, about to be read by the next transfer.
fn written_to_source() -> Barrier {
    Barrier {
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_stage: vk::PipelineStageFlags::TRANSFER,
        src_access: vk::AccessFlags::TRANSFER_WRITE,
        dst_stage: vk::PipelineStageFlags::TRANSFER,
        dst_access: vk::AccessFlags::TRANSFER_READ,
    }
}

/// Blit all of `src` (in `TRANSFER_SRC_OPTIMAL`) into all of `dst`, which is one of this
/// module's own images: its old contents are not needed, so it comes from `UNDEFINED`.
fn blit(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    src_extent: vk::Extent2D,
    dst: vk::Image,
    dst_extent: vk::Extent2D,
    flip_y: bool,
) {
    cmd_barrier(
        device,
        cmd,
        dst,
        Barrier {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_stage: vk::PipelineStageFlags::TRANSFER,
            src_access: vk::AccessFlags::empty(),
            dst_stage: vk::PipelineStageFlags::TRANSFER,
            dst_access: vk::AccessFlags::TRANSFER_WRITE,
        },
    );
    let corner = |extent: vk::Extent2D| vk::Offset3D {
        x: extent.width as i32,
        y: extent.height as i32,
        z: 1,
    };
    let src_offsets = if flip_y {
        [
            vk::Offset3D {
                x: 0,
                y: src_extent.height as i32,
                z: 0,
            },
            vk::Offset3D {
                x: src_extent.width as i32,
                y: 0,
                z: 1,
            },
        ]
    } else {
        [vk::Offset3D::default(), corner(src_extent)]
    };
    let region = vk::ImageBlit::default()
        .src_subresource(layers())
        .src_offsets(src_offsets)
        .dst_subresource(layers())
        .dst_offsets([vk::Offset3D::default(), corner(dst_extent)]);
    unsafe {
        device.cmd_blit_image(
            cmd,
            src,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
            vk::Filter::LINEAR,
        );
    }
}
