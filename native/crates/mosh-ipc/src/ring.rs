//! The shared-memory layout's accessors and its per-slot seqlock.
//!
//! A `Region` is a base pointer into a `win::Mapping`. Nothing here allocates, locks or blocks,
//! so the writer half can run in an audio callback.
//!
//! Audio: one slot per block, `IpcBlock`, 64-byte aligned. Each slot has its own seqlock (`seq`
//! odd while the writer is filling it, even when it holds a whole block). The writer publishes
//! `IpcHeader::write_index` with a release store *after* the slot is complete, so a reader that
//! sees index `i` in `write_index` is guaranteed to see a complete slot for `i` unless the writer
//! has since wrapped around onto that same slot (256 blocks later) — which the sequence re-read
//! catches.
//!
//! MIDI: `IpcMidiEvent` has no sequence of its own (the layout is frozen), so the event ring has
//! weaker protection: a 16-byte slot could be overwritten mid-copy 1024 events later. Reads
//! validate the message shape and drop nonsense instead of passing it on.

use std::mem::size_of;
use std::ptr;
use std::sync::atomic::{Ordering, fence};

use mosh_core::{
    AudioBlock, BLOCK_MAX_FRAMES, BLOCKS_OFFSET, IpcBlock, IpcHeader, IpcMidiEvent, MIDI_OFFSET,
    MIDI_RING, MidiEvent, RING_BLOCKS, Transport, TransportRaw,
};

/// A view of one mapping: header at offset 0, then the block ring, then the MIDI ring.
///
/// The base is raw so the fields that the writer mutates are never reached through a shared
/// reference (`&IpcBlock`) while another thread writes them; the atomics are the synchronisation.
pub(crate) struct Region {
    base: *mut u8,
    bytes: usize,
}

// One thread owns a `Region`; the atomics inside the region are what makes the two ends of the
// ring safe to use from two threads.
unsafe impl Send for Region {}

impl Region {
    /// # Safety
    ///
    /// `base` must point to at least `bytes` readable and writable bytes, 64-byte aligned.
    pub(crate) unsafe fn new(base: *mut u8, bytes: usize) -> Region {
        debug_assert!(
            (base as usize).is_multiple_of(64),
            "a region must be 64-byte aligned"
        );
        debug_assert!(
            bytes >= mosh_core::REGION_BYTES,
            "a region must hold the whole layout"
        );
        Region { base, bytes }
    }

    /// The header's atomics. Its plain fields are written once, at create, before anything else
    /// is published.
    pub(crate) fn header(&self) -> &IpcHeader {
        // SAFETY: the header is the first 128 bytes of the region, and lives as long as it does.
        unsafe { &*(self.base as *const IpcHeader) }
    }

    fn block_ptr(&self, slot: usize) -> *mut IpcBlock {
        debug_assert!(slot < RING_BLOCKS);
        // SAFETY: the block ring is inside the region, `slot` is in range.
        unsafe {
            self.base
                .add(BLOCKS_OFFSET + slot * size_of::<IpcBlock>())
                .cast::<IpcBlock>()
        }
    }

    fn midi_ptr(&self, slot: usize) -> *mut IpcMidiEvent {
        debug_assert!(slot < MIDI_RING);
        // SAFETY: the MIDI ring is inside the region, `slot` is in range.
        unsafe {
            self.base
                .add(MIDI_OFFSET + slot * size_of::<IpcMidiEvent>())
                .cast::<IpcMidiEvent>()
        }
    }

    /// The writer's half of `IpcWriter::create`: definition before data, then zero the rings so a
    /// region left behind by an earlier writer cannot be read as if it were current.
    pub(crate) fn initialize(&self, role: u32, sample_rate: u32, channels: u16, heartbeat_ns: u64) {
        // SAFETY: `bytes` covers header + both rings, which is what this clears.
        unsafe { ptr::write_bytes(self.base.add(BLOCKS_OFFSET), 0, self.bytes - BLOCKS_OFFSET) };

        let header = self.base.cast::<IpcHeader>();
        // SAFETY: every field is written once, before the magic makes the region visible. Raw
        // pointers throughout: a `&IpcHeader` would freeze the fields this is about to set.
        unsafe {
            (*header).sample_rate = sample_rate;
            (*header).channels = channels as u32;
            (*header).block_max_frames = BLOCK_MAX_FRAMES as u32;
            (*header).ring_blocks = RING_BLOCKS as u32;
            (*header).midi_ring = MIDI_RING as u32;
            (*header).role = role;
            (*ptr::addr_of!((*header).write_index)).store(0, Ordering::Relaxed);
            (*ptr::addr_of!((*header).midi_write_index)).store(0, Ordering::Relaxed);
            (*ptr::addr_of!((*header).dropped)).store(0, Ordering::Relaxed);
            (*ptr::addr_of!((*header).heartbeat_ns)).store(heartbeat_ns, Ordering::Relaxed);
            // Identity last: a reader that attaches mid-create sees magic 0 and reports "not ready".
            fence(Ordering::Release);
            (*header).version = mosh_core::IPC_VERSION;
            fence(Ordering::Release);
            (*header).magic = mosh_core::IPC_MAGIC;
        }
    }

    // -- writer ------------------------------------------------------------

    /// Write one whole block into slot `index % RING_BLOCKS`, then publish the index.
    /// Never allocates, never blocks. `samples` must hold at least `frames * channels` samples.
    pub(crate) fn write_block(
        &self,
        index: u64,
        transport: Option<&Transport>,
        samples: &[f32],
        frames: usize,
        channels: usize,
    ) -> bool {
        let needed = frames * channels;
        if frames == 0
            || frames > BLOCK_MAX_FRAMES
            || channels == 0
            || channels > 2
            || samples.len() < needed
        {
            return false;
        }
        let slot = self.block_ptr((index % RING_BLOCKS as u64) as usize);
        // SAFETY: `slot` is a whole `IpcBlock` inside the region; the seqlock brackets the write.
        unsafe {
            let seq = &*ptr::addr_of!((*slot).seq);
            seq.fetch_add(1, Ordering::Acquire); // odd: this slot is being written
            (*slot).frames = frames as u32;
            (*slot).channels = channels as u32;
            (*slot).transport = TransportRaw::from_transport(transport);
            ptr::copy_nonoverlapping(
                samples.as_ptr(),
                ptr::addr_of_mut!((*slot).samples).cast::<f32>(),
                needed,
            );
            seq.fetch_add(1, Ordering::Release); // even: the slot holds a whole block
        }
        // Release: the finished slot is visible to a reader that sees this index.
        self.header()
            .write_index
            .store(index + 1, Ordering::Release);
        true
    }

    /// Write one MIDI event. Never allocates, never blocks.
    pub(crate) fn write_midi(&self, index: u64, event: &MidiEvent) {
        let slot = self.midi_ptr((index % MIDI_RING as u64) as usize);
        // SAFETY: `slot` is a whole `IpcMidiEvent` inside the region; a 16-byte plain write.
        unsafe { ptr::write(slot, IpcMidiEvent::from_event(event)) };
        self.header()
            .midi_write_index
            .store(index + 1, Ordering::Release);
    }

    // -- reader ------------------------------------------------------------

    /// Read slot `index % RING_BLOCKS` into `out`, or `None` when a writer is mid-slot or the slot
    /// changed under us (torn). `out` holds no data after `None`.
    pub(crate) fn read_block(&self, index: u64, out: &mut AudioBlock) -> Option<()> {
        let slot = self.block_ptr((index % RING_BLOCKS as u64) as usize);
        // SAFETY: `slot` is a whole `IpcBlock` inside the region, bracketed by the seqlock.
        unsafe {
            let seq = &*ptr::addr_of!((*slot).seq);
            let before = seq.load(Ordering::Acquire);
            if before & 1 != 0 {
                out.samples.clear();
                out.transport = None;
                return None;
            }

            let frames = (*slot).frames as usize;
            let channels = (*slot).channels as usize;
            if frames == 0 || frames > BLOCK_MAX_FRAMES || channels == 0 || channels > 2 {
                out.samples.clear();
                out.transport = None;
                return None;
            }
            let needed = frames * channels;

            out.transport = (*slot).transport.to_transport();
            out.channels = channels as u16;
            out.samples.clear();
            let header =
                ptr::slice_from_raw_parts(ptr::addr_of!((*slot).samples).cast::<f32>(), needed);
            out.samples.extend_from_slice(&*header);

            fence(Ordering::Acquire);
            if seq.load(Ordering::Relaxed) != before {
                out.samples.clear();
                out.transport = None;
                return None;
            }
        }
        out.sample_rate = self.header().sample_rate;
        Some(())
    }

    /// Read slot `index % MIDI_RING`. A slot whose shape does not parse is reported as missing.
    pub(crate) fn read_midi(&self, index: u64) -> Option<MidiEvent> {
        let slot = self.midi_ptr((index % MIDI_RING as u64) as usize);
        // SAFETY: `slot` is a whole `IpcMidiEvent` inside the region; a 16-byte plain read.
        let raw: IpcMidiEvent = unsafe { ptr::read(slot) };
        let event = raw.to_event();
        (event.status & 0x80 != 0 && event.data1 < 0x80 && event.data2 < 0x80).then_some(event)
    }
}

#[cfg(test)]
pub(crate) fn slot_seq(region: &Region, index: u64) -> u64 {
    let slot = region.block_ptr((index % RING_BLOCKS as u64) as usize);
    // SAFETY: whole slot inside the region.
    unsafe { (*ptr::addr_of!((*slot).seq)).load(Ordering::Acquire) }
}

#[cfg(test)]
pub(crate) fn force_slot_seq(region: &Region, index: u64, value: u64) {
    let slot = region.block_ptr((index % RING_BLOCKS as u64) as usize);
    // SAFETY: whole slot inside the region; the test owns both ends.
    unsafe { (*ptr::addr_of!((*slot).seq)).store(value, Ordering::Release) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosh_core::{REGION_BYTES, ShowRole};
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    /// A zeroed, 64-byte aligned region that is not backed by a mapping: the ring logic without
    /// the Windows half.
    struct HeapRegion {
        ptr: *mut u8,
        layout: Layout,
        region: Region,
    }

    impl HeapRegion {
        fn new() -> HeapRegion {
            let layout = Layout::from_size_align(REGION_BYTES, 64).expect("layout");
            // SAFETY: a plain allocation with a fixed layout.
            let ptr = unsafe { alloc_zeroed(layout) };
            assert!(!ptr.is_null(), "region allocation failed");
            // SAFETY: `ptr` is `REGION_BYTES` long and 64-byte aligned.
            let region = unsafe { Region::new(ptr, REGION_BYTES) };
            region.initialize(ShowRole::Master as u32, 48_000, 2, 1);
            HeapRegion {
                ptr,
                layout,
                region,
            }
        }
    }

    impl Drop for HeapRegion {
        fn drop(&mut self) {
            // SAFETY: allocated with this exact layout in `new`.
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }

    #[test]
    fn torn_read_is_skipped() {
        let heap = HeapRegion::new();
        let samples = vec![0.5f32; 512];
        assert!(heap.region.write_block(0, None, &samples, 256, 2));

        let mut out = AudioBlock::default();
        assert!(
            heap.region.read_block(0, &mut out).is_some(),
            "a whole block reads back"
        );
        assert_eq!(out.frames(), 256);

        // A slot whose sequence is odd is being written: never returned, and nothing is left
        // in the caller's buffer.
        force_slot_seq(&heap.region, 0, 1);
        assert_eq!(
            heap.region.read_block(0, &mut out),
            None,
            "an odd sequence returned a block"
        );
        assert!(out.samples.is_empty(), "a skipped read left samples behind");

        // Back to a complete slot: readable again.
        force_slot_seq(&heap.region, 0, 2);
        assert!(heap.region.read_block(0, &mut out).is_some());
        assert_eq!(slot_seq(&heap.region, 0), 2);

        // Two writes bump the sequence by two each: it stays even after every write.
        assert!(heap.region.write_block(1, None, &samples, 256, 2));
        assert_eq!(slot_seq(&heap.region, 1) % 2, 0);
    }

    #[test]
    fn short_and_oversized_blocks_are_rejected() {
        let heap = HeapRegion::new();
        let samples = vec![0.0f32; 512];
        // frames * channels larger than the slice: refused, the writer would read past it.
        assert!(!heap.region.write_block(0, None, &samples[..100], 256, 2));
        // More frames than a slot can hold.
        assert!(
            !heap
                .region
                .write_block(0, None, &vec![0.0f32; 4096], 2048, 2)
        );
        // A zero-frame block is not a block.
        assert!(!heap.region.write_block(0, None, &samples, 0, 2));
        // Three interleaved channels do not fit a slot.
        assert!(!heap.region.write_block(0, None, &vec![0.0f32; 768], 256, 3));
    }
}
