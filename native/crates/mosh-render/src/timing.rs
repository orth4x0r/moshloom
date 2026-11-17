//! GPU timestamps.
//!
//! Every recorded pass is bracketed by two timestamp queries; `Gpu::wait_frame` reads them back
//! after the timeline semaphore says the frame finished. The stats overlay (Task 2.6) and the
//! budget checks in the design's performance table use `GpuTimings`.

use ash::vk;

use crate::device::Gpu;

/// Timings of one submitted frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuTimings {
    /// One entry per recorded pass, in recording order.
    pub per_pass_ms: Vec<(String, f32)>,
    /// First pass start to last pass end.
    pub frame_ms: f32,
}

/// The handle `FrameRecorder::submit` hands back. Wait for it with `Gpu::wait_frame`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameTicket {
    pub(crate) slot: usize,
    pub(crate) value: u64,
    pub(crate) passes: u32,
    pub(crate) names: Vec<String>,
}

pub(crate) fn wait_timeline(gpu: &Gpu, value: u64) -> Result<(), crate::RenderError> {
    if value == 0 {
        return Ok(());
    }
    let timeline = gpu.frames.borrow().timeline;
    let semaphores = [timeline];
    let values = [value];
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    unsafe { gpu.dev.wait_semaphores(&info, u64::MAX) }?;
    Ok(())
}

/// Reads the timestamp pairs recorded for `slot` and turns them into milliseconds.
///
/// Called after the slot's timeline value has been signalled, so the queries are complete.
pub(crate) fn read(gpu: &Gpu, ticket: &FrameTicket) -> GpuTimings {
    if ticket.passes == 0 || gpu.timestamp_valid_bits == 0 || gpu.timestamp_period_ns <= 0.0 {
        return GpuTimings::default();
    }
    let query_pool = {
        let frames = gpu.frames.borrow();
        frames.slots[ticket.slot].query_pool
    };
    let count = ticket.passes * 2;
    let mut raw = vec![0u64; count as usize];
    let result = unsafe {
        gpu.dev
            .get_query_pool_results(query_pool, 0, &mut raw, vk::QueryResultFlags::TYPE_64)
    };
    if result.is_err() {
        return GpuTimings::default();
    }

    let period_ms = gpu.timestamp_period_ns as f64 / 1_000_000.0;
    let mut per_pass_ms = Vec::with_capacity(ticket.passes as usize);
    for index in 0..ticket.passes as usize {
        let start = raw[index * 2];
        let end = raw[index * 2 + 1];
        let delta = end.saturating_sub(start);
        let name = ticket
            .names
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("pass{index}"));
        per_pass_ms.push((name, (delta as f64 * period_ms) as f32));
    }
    let span = raw[count as usize - 1].saturating_sub(raw[0]);
    GpuTimings {
        per_pass_ms,
        frame_ms: (span as f64 * period_ms) as f32,
    }
}
