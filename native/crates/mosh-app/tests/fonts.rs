//! The control window's text. egui 0.37 ships its fonts behind the `default_fonts` feature; without
//! it every character is drawn as egui's hollow "missing glyph" box (seen in the first windowed run).

#[test]
fn the_editor_can_draw_its_text() {
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
    output.textures_delta.clear();
    let text = "Session Arrangement Clip Presets Project ilanaVisual 0123456789 · BPM > scene";
    // Every character the editor writes must be in the bundled fonts (they have no U+25B6 play
    // triangle: that drew a box on the launch buttons).
    let ui_source = include_str!("../src/editor/ui.rs");
    let missing: Vec<char> = ctx.fonts_mut(|fonts| {
        let mut seen: Vec<char> = ui_source.chars().filter(|c| !c.is_ascii()).collect();
        seen.sort_unstable();
        seen.dedup();
        seen.into_iter().filter(|c| !fonts.has_glyph(&egui::FontId::default(), *c)).collect()
    });
    assert!(missing.is_empty(), "ui.rs uses characters the editor font cannot draw: {missing:?}");
    let ok = ctx.fonts_mut(|fonts| fonts.has_glyphs(&egui::FontId::default(), text));
    assert!(ok, "the editor's font is missing glyphs for {text:?}");
}
