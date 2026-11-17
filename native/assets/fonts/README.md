# Fonts for the native text layer

`mosh-visuals::text` rasterises its own glyphs at `set_text`, so it needs font bytes inside the
repository. Both faces here are freely redistributable and were copied out of a crates.io
package (no download from a font site, no font licence to negotiate):

| file | face | from | licence |
| --- | --- | --- | --- |
| `Ubuntu-Light.ttf` | Ubuntu Light (chrome, clean looks) | `epaint_default_fonts` 0.37.0, `fonts/Ubuntu-Light.ttf` | Ubuntu Font Licence 1.0 (`Ubuntu-Light-UFL.txt`) |
| `Hack-Regular.ttf` | Hack Regular (thorny / cyber look) | `epaint_default_fonts` 0.37.0, `fonts/Hack-Regular.ttf` | MIT + Bitstream Vera (`Hack-Regular-LICENSE.txt`) |

SHA-256, so a reviewer can check the bytes without trusting the crate cache:

```
80307b8da7649aa4ee4d484b232140e3ce1ec0ca093073d3c53c8f5a5ced7a70  Ubuntu-Light.ttf
15f55cc0c85a2988d2b4b3a8cdb5d77fdfbaf319e1bb5309d725db9818fb7125  Hack-Regular.ttf
```

`epaint_default_fonts` itself is `(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0`. Only the two
faces above and their licence files are copied here; its emoji faces and the icon font stay out.

## What this does not cover

There is no display face here: no Y2K bubble sans, no blackletter. The three looks in
`mosh-visuals/src/text/text.wgsl` are built out of these two faces plus shader styling (SDF
dilation for the chrome bubble, noise-warped edges and thorns for the cyber look). Adding a real
display face is a coordinator decision — it needs a licence file next to it, the same SHA-256
note, and a look selection that actually uses it.

## Display faces (added by the coordinator, 2026-11-17)

Fetched from the google/fonts repository (`ofl/<family>/`), each under the SIL Open Font License
1.1 (licence text next to the font). Variable fonts; the text layer uses their default instance.

| family | file | use | sha256 |
| --- | --- | --- | --- |
| Grenze Gotisch | `grenzegotisch/GrenzeGotisch[wght].ttf` | blackletter, the thorny / sigil look | 701b299d8dc002a2… |
| Unbounded | `unbounded/Unbounded[wght].ttf` | wide rounded display, the chrome look | 323b511be380c8d4… |
| Space Grotesk | `spacegrotesk/SpaceGrotesk[wght].ttf` | clean grotesque | acad6de1fc93436f… |
