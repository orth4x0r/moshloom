# Fonts for the native text layer

`mosh-visuals::text` rasterises its own glyphs at `set_text`, so it needs font bytes inside the
repository. Every face here is freely redistributable and sits next to its licence file.

## The three looks and their faces

`p1` ("font") picks a look, and each look owns one display face from this directory:

| `p1` | look | file (this directory) | family | licence |
| --- | --- | --- | --- | --- |
| 0 | Chrome | `unbounded/Unbounded[wght].ttf` | Unbounded | SIL OFL 1.1 (`unbounded/OFL.txt`) |
| 1 | Sigil | `grenzegotisch/GrenzeGotisch[wght].ttf` | Grenze Gotisch | SIL OFL 1.1 (`grenzegotisch/OFL.txt`) |
| 2 | Clean | `spacegrotesk/SpaceGrotesk[wght].ttf` | Space Grotesk | SIL OFL 1.1 (`spacegrotesk/OFL.txt`) |

They were fetched from the google/fonts repository (`ofl/<family>/`). All three are **variable**
fonts; `fontdue` 0.9 rasterises the font's default instance (no `fvar` support in the parser), so
the shapes are the regular weight. The text layer makes up the missing optical weight with a small
field dilation per look (`text/text.wgsl`, `bold`), which is also what keeps the hairline default
instances of Grenze Gotisch and Space Grotesk readable as display type.

## Inside the repository

The three faces and their licences are the only files copied here; nothing is downloaded at build
time. The text layer includes them with `include_bytes!` from `src/text.rs` (`FACE_CHROME`,
`FACE_SIGIL`, `FACE_CLEAN`).

## Fallbacks

`Ubuntu-Light.ttf` (Ubuntu Font Licence 1.0, `Ubuntu-Light-UFL.txt`) and `Hack-Regular.ttf` (MIT +
Bitstream Vera, `Hack-Regular-LICENSE.txt`) came from `epaint_default_fonts` 0.37.0. Neither look
uses them any more; they stay because they are small, freely redistributable, and still parse if a
display face is ever missing. The layer draws black rather than failing to start when a face
cannot be parsed.

SHA-256, so a reviewer can check the bytes without trusting the crate cache:

```
323b511be380c8d474ef030686b71aedde501f8d9cd46da558b7c40454372c3f  unbounded/Unbounded[wght].ttf
701b299d8dc002a2b4bea2ff0f1272c0e4081a2835914354804565c410d0c637  grenzegotisch/GrenzeGotisch[wght].ttf
acad6de1fc93436f5c0f1f4137751ef04f1aea3063e7036535970ffcfbd79f72  spacegrotesk/SpaceGrotesk[wght].ttf
80307b8da7649aa4ee4d484b232140e3ce1ec0ca093073d3c53c8f5a5ced7a70  Ubuntu-Light.ttf
15f55cc0c85a2988d2b4b3a8cdb5d77fdfbaf319e1bb5309d725db9818fb7125  Hack-Regular.ttf
```

`epaint_default_fonts` itself is `(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0`. Only the two
faces above and their licence files are copied here; its emoji faces and the icon font stay out.
