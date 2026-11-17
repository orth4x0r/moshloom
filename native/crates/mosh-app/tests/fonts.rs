//! The control window's text. egui 0.37 ships its fonts behind the `default_fonts` feature; without
//! it every character is drawn as egui's hollow "missing glyph" box (seen in the first windowed run).

#[test]
fn the_editor_can_draw_its_text() {
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
    output.textures_delta.clear();
    let text = "Session Arrangement Follow Manual Autopilot ilanaVisual ILANA AUDIO 0123456789 % + - ? . () · BPM > scene";
    // Every character the editor writes must be in the bundled fonts (they have no U+25B6 play
    // triangle: that drew a box on the launch buttons).
    // Every file that puts text in the control window: the views, the theme, the model's
    // messages, and the catalog's names (which other lanes add to).
    let sources = [
        include_str!("../src/editor/ui/mod.rs"),
        include_str!("../src/editor/ui/topbar.rs"),
        include_str!("../src/editor/ui/browser.rs"),
        include_str!("../src/editor/ui/session.rs"),
        include_str!("../src/editor/ui/arrangement.rs"),
        include_str!("../src/editor/ui/detail.rs"),
        include_str!("../src/editor/ui/widgets.rs"),
        include_str!("../src/editor/theme.rs"),
        include_str!("../src/editor/model.rs"),
        include_str!("../src/core.rs"),
    ];
    let mut ui_source: String = sources.concat();
    for entry in mosh_app::editor::model::catalog() {
        ui_source.push_str(entry.name);
        ui_source.push_str(entry.group);
        for param in entry.params {
            ui_source.push_str(param);
        }
    }
    for (title, rows) in mosh_app::editor::ui::SHORTCUTS {
        ui_source.push_str(title);
        for (keys, what) in rows {
            ui_source.push_str(keys);
            ui_source.push_str(what);
        }
    }
    let missing: Vec<char> = ctx.fonts_mut(|fonts| {
        let mut seen: Vec<char> = ui_source
            .chars()
            .filter(|c| !c.is_ascii() && !c.is_control())
            // Box-drawing and arrows appear only in doc comments (layout sketches), never in a
            // string the window draws.
            .filter(|c| !('\u{2190}'..='\u{25ff}').contains(c))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen.into_iter().filter(|c| !fonts.has_glyph(&egui::FontId::default(), *c)).collect()
    });
    assert!(
        missing.is_empty(),
        "the control window uses characters the editor font cannot draw: {missing:?}"
    );
    let ok = ctx.fonts_mut(|fonts| fonts.has_glyphs(&egui::FontId::default(), text));
    assert!(ok, "the editor's font is missing glyphs for {text:?}");
}
