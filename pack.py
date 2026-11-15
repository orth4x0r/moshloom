import base64, pathlib, sys

root = pathlib.Path(__file__).resolve().parent


def read(name):
    p = root / name
    if not p.exists():
        sys.exit(f"pack.py: missing {name} - run `npm install` first" if 'node_modules' in name else f"pack.py: missing {name}")
    return p.read_text(encoding='utf-8')


mux = (read('node_modules/mp4-muxer/build/mp4-muxer.js') + "\n"
       + read('node_modules/webm-muxer/build/webm-muxer.js'))
assert '</script' not in mux.lower(), "a muxer contains '</script' and cannot be inlined"
s = read('index.src.html')
s = (s.replace('__MUX__', mux)
      .replace('__GPU__', read('gpu.js'))
      .replace('__WASM__', base64.b64encode((root / 'engine.wasm').read_bytes()).decode('ascii')))
(root / 'index.html').write_text(s, encoding='utf-8')
print(f"index.html {len(s)} bytes")
