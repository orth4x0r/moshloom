// Applies the S1 spike to gpu.js (throwaway). Revert with: copy gpu.js.bak gpu.js
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '../..');
const gpuPath = path.join(root, 'gpu.js');
const bakPath = path.join(root, 'gpu.js.bak');
if (!fs.existsSync(bakPath)) { console.error('no gpu.js.bak - refusing to patch'); process.exit(1); }

let s = fs.readFileSync(gpuPath, 'utf8');
const block = fs.readFileSync(path.join(root, 'lab/spike/block.js'), 'utf8')
  .split('\n').slice(2).join('\n');           // drop the two comment lines

const must = (a) => { if (!s.includes(a)) { console.error('anchor missing:\n' + a.slice(0, 80)); process.exit(1); } };

const SCENE_ANCHOR = 'const G={ok:false,';
must(SCENE_ANCHOR);
s = s.replace(SCENE_ANCHOR, block + '\n' + SCENE_ANCHOR);

const INIT_ANCHOR = '  G.fluidOn=false;\n  assertLayouts();';
must(INIT_ANCHOR);
s = s.replace(INIT_ANCHOR, `  G.fluidOn=false;
  // S1 spike (throwaway)
  G.spike=true; G.rmSteps=64;
  const sp=await Promise.all([WG_RM_SPIKE(40),WG_RM_SPIKE(64)].map((c,i)=>G.mod(c,'spike '+i)));
  G.pSpike=await Promise.all(sp.map(m=>G.pipe(m,'rgba16float',G.sceneLayout)));
  G.pUp=await G.pipe(await G.mod(WG_UPSCALE,'upscale'),'rgba16float','auto');
  G.half=G.tex(640,360,'rgba16float',RT);
  assertLayouts();`);

const PASS_ANCHOR = `  if(xf<0) gpuScenePass(enc,tgt.v,scA,G.uA,pl[20],pl[21]);`;
must(PASS_ANCHOR);
s = s.replace(PASS_ANCHOR, `  if(xf<0&&G.spike&&scA===0){                    // S1: half-res march, then nearest upscale
    if(G.half.width!==(G.w>>1)){ G.half.destroy(); G.half=G.tex(G.w>>1,G.h>>1,'rgba16float',GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT); }
    gpuSU(G.uA,pl,16,f,G.w>>1,G.h>>1);
    const p=enc.beginRenderPass({colorAttachments:[{view:G.half.v,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
    p.setPipeline(G.rmSteps===40?G.pSpike[0]:G.pSpike[1]);
    p.setBindGroup(0,dev.createBindGroup({layout:G.sceneBGL,entries:[{binding:0,resource:{buffer:G.uA}},{binding:1,resource:G.lin},{binding:2,resource:G.palT.v},{binding:3,resource:G.caT.v},
      {binding:4,resource:G.melT.v},{binding:5,resource:G.rd[G.ri].v},{binding:6,resource:G.blank.v}]}));
    p.draw(3); p.end();
    full(tgt.v,G.pUp,bg(G.pUp,[G.half.v]),'scene');
  } else if(xf<0) gpuScenePass(enc,tgt.v,scA,G.uA,pl[20],pl[21]);`);

fs.writeFileSync(gpuPath, s);
console.log('spike applied to gpu.js');
