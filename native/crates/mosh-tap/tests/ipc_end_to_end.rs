//! The plugin's real sink, written through shared memory and read back by the app's reader.
//! Uses the `other` role so it cannot collide with a Tap running in Live on the master.

#![cfg(feature = "ipc")]

use mosh_core::ipc_layout::ShowRole;
use mosh_core::transport::{MidiEvent, Transport};
use mosh_ipc::IpcReader;
use mosh_tap::sink::{BlockSink, PluginSink};

#[test]
fn plugin_sink_reaches_the_app_reader() {
    let mut sink = PluginSink::create(ShowRole::Other, 48_000, 2);
    assert!(
        sink.is_connected(),
        "the real shared-memory writer did not open"
    );
    let mut reader = IpcReader::open(ShowRole::Other).expect("reader opens the plugin's region");

    let t = Transport {
        tempo_bpm: 93.0,
        ppq: 6.5,
        bar_start_ppq: 4.0,
        sig_num: 4,
        sig_den: 4,
        playing: true,
        sample_pos: 512,
    };
    let samples: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
    sink.push_block(Some(&t), &samples, 256);
    sink.push_midi(&MidiEvent {
        ppq: 6.5,
        status: 0x90,
        data1: 36,
        data2: 100,
    });

    let (mut blocks, mut midi) = (Vec::new(), Vec::new());
    let stats = reader.poll(&mut blocks, &mut midi);
    assert_eq!(stats.new_blocks, 1);
    assert!(stats.writer_alive);
    assert_eq!(blocks[0].transport, Some(t));
    assert_eq!(blocks[0].samples, samples);
    assert_eq!(
        midi,
        vec![MidiEvent {
            ppq: 6.5,
            status: 0x90,
            data1: 36,
            data2: 100
        }]
    );
}
