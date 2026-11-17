//! Render targets: the per-layer ping-pong pair (RGBA16F) that feedback reads from and writes
//! to, and the output image the compositor draws into.
//!
//! Every image is kept in `SHADER_READ_ONLY_OPTIMAL` between passes, so a target can be sampled
//! by the next draw, by the compositor, or read back without extra bookkeeping. A pass moves its
//! write image to `COLOR_ATTACHMENT_OPTIMAL` and back.

use std::cell::Cell;

use ash::vk;

use crate::device::{create_buffer, Gpu};

/// Layer targets are half float: a feedback chain that adds a few percent per frame bands badly
/// in 8 bits, and the compositor has the bandwidth for it at these resolutions.
pub const OFFSCREEN_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// One image layout transition, expressed the way the passes here use it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Barrier {
    pub(crate) old_layout: vk::ImageLayout,
    pub(crate) new_layout: vk::ImageLayout,
    pub(crate) src_stage: vk::PipelineStageFlags,
    pub(crate) src_access: vk::AccessFlags,
    pub(crate) dst_stage: vk::PipelineStageFlags,
    pub(crate) dst_access: vk::AccessFlags,
}

pub(crate) fn cmd_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    barrier: Barrier,
) {
    let image_barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(barrier.src_access)
        .dst_access_mask(barrier.dst_access)
        .old_layout(barrier.old_layout)
        .new_layout(barrier.new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            barrier.src_stage,
            barrier.dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[image_barrier],
        );
    }
}

/// Barriers for the two shapes this crate needs.
pub(crate) fn to_color_attachment() -> Barrier {
    Barrier {
        old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    }
}

pub(crate) fn to_shader_read() -> Barrier {
    Barrier {
        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
        dst_access: vk::AccessFlags::SHADER_READ,
    }
}

/// A readable image about to be written by a transfer (`vkCmdClearColorImage` recorded in the
/// frame, instead of a one-shot submission that the CPU waits on).
pub(crate) fn to_transfer_dst() -> Barrier {
    Barrier {
        old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage: vk::PipelineStageFlags::TRANSFER,
        dst_access: vk::AccessFlags::TRANSFER_WRITE,
    }
}

/// The same transfer, back to readable.
pub(crate) fn from_transfer_write() -> Barrier {
    Barrier {
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_stage: vk::PipelineStageFlags::TRANSFER,
        src_access: vk::AccessFlags::TRANSFER_WRITE,
        dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
        dst_access: vk::AccessFlags::SHADER_READ,
    }
}

struct ImagePair {
    images: [vk::Image; 2],
    memories: [vk::DeviceMemory; 2],
    views: [vk::ImageView; 2],
}

/// A layer's render target: two RGBA16F images that alternate, so a fragment shader can read
/// the previous frame (feedback, mosh, motion) while writing the current one.
pub struct LayerTarget {
    device: ash::Device,
    pair: ImagePair,
    layouts: [Cell<vk::ImageLayout>; 2],
    front: Cell<usize>,
    width: u32,
    height: u32,
}

impl LayerTarget {
    /// Creates the ping-pong pair, cleared to transparent black.
    ///
    /// Infallible by contract (Task 1.7 Interfaces): a device that cannot allocate a layer
    /// target cannot render the show either, so this panics rather than threading a `Result`
    /// through every caller. Layer sizes come from the project, not from user input.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> LayerTarget {
        assert!(width > 0 && height > 0, "layer target needs a non-zero size");
        let extent = vk::Extent2D { width, height };
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;

        let mut images = [vk::Image::null(); 2];
        let mut memories = [vk::DeviceMemory::null(); 2];
        let mut views = [vk::ImageView::null(); 2];
        for index in 0..2 {
            let (image, memory) = gpu
                .create_image(extent, OFFSCREEN_FORMAT, usage)
                .expect("allocate layer target image");
            let view = gpu
                .create_image_view(image, OFFSCREEN_FORMAT)
                .expect("create layer target view");
            images[index] = image;
            memories[index] = memory;
            views[index] = view;
        }

        let layouts = [
            Cell::new(vk::ImageLayout::UNDEFINED),
            Cell::new(vk::ImageLayout::UNDEFINED),
        ];
        clear_pair(gpu, &images);
        for layout in &layouts {
            layout.set(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }

        LayerTarget {
            device: gpu.dev.clone(),
            pair: ImagePair {
                images,
                memories,
                views,
            },
            layouts,
            front: Cell::new(0),
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> vk::Format {
        OFFSCREEN_FORMAT
    }

    pub(crate) fn front_image(&self) -> vk::Image {
        self.pair.images[self.front.get()]
    }

    pub(crate) fn front_view(&self) -> vk::ImageView {
        self.pair.views[self.front.get()]
    }

    pub(crate) fn front_layout(&self) -> vk::ImageLayout {
        self.layouts[self.front.get()].get()
    }

    pub(crate) fn back_image(&self) -> vk::Image {
        self.pair.images[1 - self.front.get()]
    }

    pub(crate) fn back_view(&self) -> vk::ImageView {
        self.pair.views[1 - self.front.get()]
    }

    pub(crate) fn set_back_layout(&self, layout: vk::ImageLayout) {
        self.layouts[1 - self.front.get()].set(layout);
    }

    /// Makes the image just written the readable one.
    pub(crate) fn swap(&self) {
        self.front.set(1 - self.front.get());
    }
}

impl Drop for LayerTarget {
    fn drop(&mut self) {
        unsafe {
            for index in 0..2 {
                self.device.destroy_image_view(self.pair.views[index], None);
                self.device.destroy_image(self.pair.images[index], None);
                self.device.free_memory(self.pair.memories[index], None);
            }
        }
    }
}

/// The compositor's destination, and the stage's source: one RGBA16F image, sized like the
/// output. `Stage` owns one at swapchain size; tests own small ones.
pub struct OutputImage {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    layout: Cell<vk::ImageLayout>,
    width: u32,
    height: u32,
}

impl OutputImage {
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> OutputImage {
        assert!(width > 0 && height > 0, "output image needs a non-zero size");
        let extent = vk::Extent2D { width, height };
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let (image, memory) = gpu
            .create_image(extent, OFFSCREEN_FORMAT, usage)
            .expect("allocate output image");
        let view = gpu
            .create_image_view(image, OFFSCREEN_FORMAT)
            .expect("create output view");
        clear_pair(gpu, &[image]);
        OutputImage {
            device: gpu.dev.clone(),
            image,
            memory,
            view,
            layout: Cell::new(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub(crate) fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.set(layout);
    }
}

impl Drop for OutputImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

fn clear_pair(gpu: &Gpu, images: &[vk::Image]) {
    let clear = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 0.0],
    };
    gpu.one_shot(|cmd| {
        for image in images {
            cmd_barrier(
                &gpu.dev,
                cmd,
                *image,
                Barrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                    src_access: vk::AccessFlags::empty(),
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                    dst_access: vk::AccessFlags::TRANSFER_WRITE,
                },
            );
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            unsafe {
                gpu.dev.cmd_clear_color_image(
                    cmd,
                    *image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear,
                    &[range],
                );
            }
            cmd_barrier(
                &gpu.dev,
                cmd,
                *image,
                Barrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::TRANSFER,
                    src_access: vk::AccessFlags::TRANSFER_WRITE,
                    dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_access: vk::AccessFlags::SHADER_READ,
                },
            );
        }
    })
    .expect("clear render target");
}

impl Gpu {
    /// Reads a layer target's current image back as tightly packed RGBA8, top row first.
    ///
    /// The target is left readable (`SHADER_READ_ONLY_OPTIMAL`), so this is safe to call between
    /// frames; it waits for its own submission.
    pub fn read_back(&self, target: &LayerTarget) -> Vec<u8> {
        let image = target.front_image();
        let layout = target.layouts[target.front.get()].get();
        let bytes = self.read_image_rgba16f(image, layout, target.width, target.height);
        target.layouts[target.front.get()].set(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        bytes
    }

    /// Reads the compositor's output back as tightly packed RGBA8.
    pub fn read_back_output(&self, out: &OutputImage) -> Vec<u8> {
        let bytes = self.read_image_rgba16f(out.image, out.layout.get(), out.width, out.height);
        out.layout.set(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        bytes
    }

    fn read_image_rgba16f(
        &self,
        image: vk::Image,
        layout: vk::ImageLayout,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let texels = u64::from(width) * u64::from(height);
        let bytes = texels * 8;
        let (buffer, memory) = create_buffer(
            &self.dev,
            &self.memory,
            bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("allocate readback buffer");

        self.one_shot(|cmd| {
            if layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                cmd_barrier(
                    &self.dev,
                    cmd,
                    image,
                    Barrier {
                        old_layout: layout,
                        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
                            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        src_access: vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_stage: vk::PipelineStageFlags::TRANSFER,
                        dst_access: vk::AccessFlags::TRANSFER_READ,
                    },
                );
            }
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            unsafe {
                self.dev.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer,
                    &[region],
                );
            }
            cmd_barrier(
                &self.dev,
                cmd,
                image,
                Barrier {
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::TRANSFER,
                    src_access: vk::AccessFlags::TRANSFER_READ,
                    dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_access: vk::AccessFlags::SHADER_READ,
                },
            );
        })
        .expect("copy image to readback buffer");

        let pointer = unsafe {
            self.dev
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
                .expect("map readback buffer")
        };
        let halves = unsafe { std::slice::from_raw_parts(pointer as *const u16, (texels * 4) as usize) };
        let mut out = vec![0u8; (texels * 4) as usize];
        for (destination, half) in out.iter_mut().zip(halves.iter()) {
            let value = half_to_f32(*half).clamp(0.0, 1.0);
            *destination = (value * 255.0).round() as u8;
        }
        unsafe {
            self.dev.unmap_memory(memory);
            self.dev.destroy_buffer(buffer, None);
            self.dev.free_memory(memory, None);
        }
        out
    }
}

/// IEEE 754 half -> single. Written out instead of pulling in `half` for one function.
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;
    let value = if exponent == 0 {
        (mantissa as f32) * 2f32.powi(-24)
    } else if exponent == 0x1f {
        if mantissa == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mantissa as f32 / 1024.0) * 2f32.powi(i32::from(exponent) - 15)
    };
    if sign == 1 { -value } else { value }
}

#[cfg(test)]
mod tests {
    use super::half_to_f32;

    #[test]
    fn half_to_f32_matches_known_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert!((half_to_f32(0x33d6) - 0.244_873).abs() < 1e-6);
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
    }
}
