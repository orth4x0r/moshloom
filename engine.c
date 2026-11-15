// MOSHLOOM engine — audio analysis, procedural scenes, datamosh feedback, transitions.
// Freestanding C -> wasm32 (no libc).  Build: see build.sh
typedef unsigned char u8; typedef unsigned int u32; typedef unsigned long size_t;
#define EXPORT __attribute__((visibility("default")))
#define MAXW 1280
#define MAXH 720
#define MAXP (MAXW*MAXH)
#define MAXF 21600          /* 12 min @ 30 fps */
#define MAXB 512
#define FPS 30.0f
#define NPAL 8
#define GW 320
#define GH 180
#define BW 40
#define BH 23

void *memset(void *d, int c, size_t n){ u8 *p=d; while(n--) *p++=(u8)c; return d; }
void *memcpy(void *d, const void *s, size_t n){ u8 *p=d; const u8 *q=s; while(n--) *p++=*q++; return d; }
void *memmove(void *d, const void *s, size_t n){ u8 *p=d; const u8 *q=s; if(p<q) while(n--) *p++=*q++; else { p+=n; q+=n; while(n--) *--p=*--q; } return d; }

// ------------------------------------------------------------------ math
#define PI 3.14159265358979
static double d_sin(double x){ x -= 2*PI*(double)(long long)(x/(2*PI)); if(x>PI) x-=2*PI; if(x<-PI) x+=2*PI;
  double t=x, s=x, x2=x*x; for(int i=1;i<12;i++){ t*=-x2/((2*i)*(2*i+1)); s+=t; } return s; }
static double d_exp(double x){ int n=(int)(x/0.6931471805599453 + (x<0?-0.5:0.5)); double r=x-n*0.6931471805599453, t=1, s=1;
  for(int i=1;i<16;i++){ t*=r/i; s+=t; } while(n>0){ s*=2; n--; } while(n<0){ s*=0.5; n++; } return s; }
static double d_log(double x){ int e=0; while(x>1.5){ x*=0.5; e++; } while(x<0.75){ x*=2; e--; }
  double y=(x-1)/(x+1), y2=y*y, t=y, s=0; for(int i=1;i<40;i+=2){ s+=t/i; t*=y2; } return 2*s+e*0.6931471805599453; }
static double d_sqrt(double x){ return __builtin_sqrt(x); }
static double d_atan(double x){ int inv=0, neg=0; if(x<0){ x=-x; neg=1; } if(x>1){ x=1/x; inv=1; }
  x=x/(1+d_sqrt(1+x*x)); x=x/(1+d_sqrt(1+x*x)); double x2=x*x, t=x, s=0; for(int i=1;i<24;i+=2){ s+=t/i; t*=-x2; }
  s*=4; if(inv) s=PI/2-s; return neg?-s:s; }
static double d_atan2(double y, double x){ if(x>0) return d_atan(y/x); if(x<0) return d_atan(y/x)+(y>=0?PI:-PI); return y>0?PI/2:(y<0?-PI/2:0); }

static float SINT[4096];
static inline float fsin(float x){ return SINT[((int)(x*651.8986469f))&4095]; }
static inline float fcos(float x){ return SINT[((int)(x*651.8986469f)+1024)&4095]; }
static inline float fsqrt(float x){ return __builtin_sqrtf(x); }
static inline float ffloor(float x){ return __builtin_floorf(x); }
static inline float fabs_(float x){ return x<0?-x:x; }
static inline float fmin_(float a,float b){ return a<b?a:b; }
static inline float fmax_(float a,float b){ return a>b?a:b; }
static inline float clamp01(float x){ return x<0?0:(x>1?1:x); }
static inline float flog(float x){ union{float f;u32 i;}u; u.f=x; float e=(float)((int)(u.i>>23)-127); u.i=(u.i&0x7fffff)|0x3f800000;
  float m=u.f; return (e+(-0.33582878f*m+2.0f)*m-0.65871759f)*0.69314718f; }
static u32 rs=1;
static inline u32 rnd(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }

// ------------------------------------------------------------------ heap (audio upload)
extern u8 __heap_base;
static size_t heap_top=0;
EXPORT void *alloc(int n){ if(!heap_top) heap_top=(size_t)&__heap_base; heap_top=(heap_top+15)&~(size_t)15;
  size_t need=heap_top+(size_t)n, have=(size_t)__builtin_wasm_memory_size(0)*65536;
  if(need>have){ if(__builtin_wasm_memory_grow(0,(need-have+65535)/65536)==(size_t)-1) return 0; }
  void *p=(void*)heap_top; return p; }   /* single live allocation: each call reuses the heap */

// ------------------------------------------------------------------ state
static int W=640,H=360,NP=640*360,SS=1;      // SS = W/640
static float GXa[MAXP],GYa[MAXP],GRa[MAXP],GTa[MAXP],VIG[MAXP];
static float bufA[MAXP*3],bufB[MAXP*3],bufC[MAXP*3],bufD[MAXP*3];
static float *B=bufA,*Tm=bufB,*S=bufC,*Pv=bufD;
static u8 OUT[MAXP*4];
static u8 grayA[GW*GH],grayB[GW*GH]; static u8 *G=grayA,*PG=grayB;
static float mvx[BW*BH],mvy[BW*BH];
static u8 CA[3][512*512];
static float LUT[NPAL][1024*3];
static float PLINE[NPAL][3][3];

// features (per 30fps frame)
static int NF=0; static float BPM=120, OFFS=0;
static float fBass[MAXF],fMid[MAXF],fHigh[MAXF],fRms[MAXF],fKick[MAXF],fSnare[MAXF],fHat[MAXF],fOn[MAXF];
static float fMel[MAXF*24],fPulse[MAXF],fTT[MAXF],fHue[MAXF];

typedef struct { float start,len,scene,pal,trans,tlen,speed,mosh,strobe,glitch,rgb,var,dens,bloom,ms0,md0,ma0,ms1,md1,ma1,ms2,md2,ma2,hueo; } Blk;   /* 24 floats */
typedef struct { float start,len,type,amt,par; } Fx;
static Fx FXB[256]; static int NFX=0;
static Blk BL[MAXB]; static int NB=0;
static int lastK=-1,lastF=-100,lastSel=0,needI=1,lastFlashF=-100000;   /* lastFlashF: frame of last full-frame flash (photosensitivity limit, >=12 frames apart) */

// per-draw audio context
static float aB,aM,aH,aP,aHue; static const float *aMel; static int aVar,aF; static const float *LUTp; static int PALi;
static inline const float *lut(float v){ return LUTp+(((int)(v*1024.f))&1023)*3; }

// ------------------------------------------------------------------ init
static const float KEYS[NPAL][8][3]={
 {{0,0,0}},
 {{0,0,0},{1,.02f,.05f},{1,.45f,0},{0,0,0},{.1f,1,.15f},{0,0,0},{.6f,.05f,1},{1,1,1}},          // 1 NOE
 {{0,0,0},{.1f,1,.35f},{0,.12f,.05f},{0,0,0},{1,.05f,.7f},{0,0,0}},                              // 2 CCRU
 {{.05f,.02f,.2f},{.85f,.85f,.95f},{1,.25f,.6f},{0,0,0},{.2f,.9f,.8f},{.1f,.05f,.3f}},           // 3 PLANT
 {{0,0,0},{.35f,0,.02f},{1,.05f,.05f},{1,.5f,.1f},{1,.75f,.7f},{.5f,0,.2f}},                     // 4 FLESH
 {{0,0,0},{1,1,1},{0,0,0},{.45f,.45f,.45f},{0,0,0},{1,1,1}},                                     // 5 MONO
 {{0,0,0},{.05f,.05f,.6f},{.5f,.1f,1},{0,0,0},{.1f,.9f,1},{1,1,1}},                              // 6 UV
 {{0,0,0},{1,.95f,0},{.4f,1,0},{0,0,0},{1,.1f,.6f},{.1f,0,.15f}}};                               // 7 ACID
static const int KEYN[NPAL]={1,8,6,6,6,6,6,6};
static const float PL0[NPAL][3][3]={ {{0}},
 {{1,.05f,.05f},{.1f,1,.15f},{.6f,.05f,1}}, {{.35f,1,.12f},{1,.06f,.7f},{0,.47f,.63f}}, {{.9f,.9f,1},{1,.25f,.6f},{.2f,.9f,.8f}},
 {{1,.05f,.05f},{1,.5f,.1f},{1,.7f,.7f}}, {{1,1,1},{.5f,.5f,.5f},{1,1,1}}, {{.5f,.1f,1},{.1f,.9f,1},{.1f,.2f,1}}, {{1,.95f,0},{1,.1f,.6f},{.4f,1,0}} };

static int inited=0;
static void init_tables(void){
  for(int i=0;i<4096;i++) SINT[i]=(float)d_sin(i*(2*PI/4096));
  for(int p=1;p<NPAL;p++){ int k=KEYN[p];
    for(int i=0;i<1024;i++){ float v=i/1024.f*k; int a=(int)v; float f=v-a; f=f*f*(3-2*f); int b=(a+1)%k;
      for(int c=0;c<3;c++) LUT[p][i*3+c]=KEYS[p][a][c]*(1-f)+KEYS[p][b][c]*f; }
    for(int j=0;j<3;j++) for(int c=0;c<3;c++) PLINE[p][j][c]=PL0[p][j][c]; }
  static const int rules[3]={110,30,90}; u32 s=12345;
  for(int r=0;r<3;r++){ u8 row[512],nx[512]; for(int i=0;i<512;i++){ s=s*1664525u+1013904223u; row[i]=(s>>16)&1; }
    for(int j=0;j<512;j++){ for(int i=0;i<512;i++){ CA[r][j*512+i]=row[i]; int v=(row[(i+511)&511]<<2)|(row[i]<<1)|row[(i+1)&511]; nx[i]=(rules[r]>>v)&1; }
      for(int i=0;i<512;i++) row[i]=nx[i]; } }
  inited=1;
}
EXPORT void reset(void){ needI=1; lastK=-1; lastF=-100; lastSel=0; lastFlashF=-100000; }
EXPORT void init(int w,int h){
  if(!inited) init_tables();
  if(w>MAXW) w=MAXW; if(h>MAXH) h=MAXH; W=w; H=h; NP=w*h; SS=w/640; if(SS<1) SS=1;
  for(int y=0;y<H;y++) for(int x=0;x<W;x++){ int p=y*W+x; float X=(x+0.5f-W*0.5f)/(H*0.5f), Y=(y+0.5f-H*0.5f)/(H*0.5f);
    GXa[p]=X; GYa[p]=Y; GRa[p]=fsqrt(X*X+Y*Y)+1e-4f; GTa[p]=(float)d_atan2(Y,X);
    float vx=(x-W*0.5f)/(W*0.5f), vy=(y-H*0.5f)/(H*0.5f), q=vx*vx+vy*vy; if(q>1.6f) q=1.6f; VIG[p]=1-0.45f*q; }
  memset(bufA,0,sizeof(float)*NP*3); memset(bufD,0,sizeof(float)*NP*3); reset();
}
EXPORT u8 *frame_ptr(void){ return OUT; }
EXPORT float *blocks_ptr(void){ return (float*)BL; }
EXPORT int block_floats(void){ return (int)(sizeof(Blk)/sizeof(float)); }   /* record size source of truth */
EXPORT void set_nblocks(int n){ NB=n>MAXB?MAXB:n; }
EXPORT float *fx_ptr(void){ return (float*)FXB; }
EXPORT int fx_floats(void){ return (int)(sizeof(Fx)/sizeof(float)); }
EXPORT void set_nfx(int n){ NFX=n>256?256:n; }
EXPORT int get_nf(void){ return NF; }
EXPORT void load_features(int n){ if(n<0)n=0; if(n>MAXF)n=MAXF; NF=n; }   /* the render side loads analysis made elsewhere */
EXPORT float get_bpm(void){ return BPM; }
EXPORT float get_offs(void){ return OFFS; }
EXPORT float *feat(int id){ switch(id){ case 0:return fRms; case 1:return fBass; case 2:return fMid; case 3:return fHigh; case 4:return fKick; case 5:return fSnare; case 6:return fHat; default:return fMel; } }

// ------------------------------------------------------------------ audio analysis
static float fre[2048],fim[2048],twr[1024],twi[1024],win[2048],pmag[1024]; static int fft_ok=0;
static void fft(void){ const int n=2048;
  for(int i=1,j=0;i<n;i++){ int bit=n>>1; for(;j&bit;bit>>=1) j^=bit; j^=bit; if(i<j){ float t=fre[i]; fre[i]=fre[j]; fre[j]=t; t=fim[i]; fim[i]=fim[j]; fim[j]=t; } }
  for(int len=2;len<=n;len<<=1){ int st=n/len; for(int i=0;i<n;i+=len) for(int k=0;k<len/2;k++){ float wr=twr[k*st], wi=twi[k*st];
      int a=i+k,b=a+len/2; float xr=fre[b]*wr-fim[b]*wi, xi=fre[b]*wi+fim[b]*wr; fre[b]=fre[a]-xr; fim[b]=fim[a]-xi; fre[a]+=xr; fim[a]+=xi; } } }
static float scratch[MAXF*24];
static float pctl(const float *x,int n,int stride,float q){ if(n<=0) return 1; int m=0; for(int i=0;i<n;i+=stride) scratch[m++]=x[i];
  for(int gap=m/2;gap>0;gap/=2) for(int i=gap;i<m;i++){ float t=scratch[i]; int j=i; for(;j>=gap&&scratch[j-gap]>t;j-=gap) scratch[j]=scratch[j-gap]; scratch[j]=t; }
  float v=scratch[(int)(q*(m-1))]; return v>1e-9f?v:1e-9f; }
static void norm(float *x,int n,float q){ float p=pctl(x,n,1,q); for(int i=0;i<n;i++){ float v=x[i]/p; x[i]=v>1.5f?1.5f:v; } }
static void envf(float *x,int n,float att,float rel){ float v=0; for(int i=0;i<n;i++){ v+=(x[i]-v)*(x[i]>v?att:rel); x[i]=v; } }
static float onset_at(float p){ int i=(int)p; if(i<0||i+1>=NF) return 0; float f=p-i; return fOn[i]*(1-f)+fOn[i+1]*f; }
static float comb(float bpm,float *bestph,float phstep){ float T=FPS*60/bpm, best=-1, bp=0;
  for(float ph=0;ph<T;ph+=phstep){ float s=0; int c=0; for(float t=ph;t<NF-1;t+=T){ s+=onset_at(t); c++; } if(c) s/=c; if(s>best){ best=s; bp=ph; } }
  *bestph=bp; return best; }
EXPORT float dbg_comb(float bpm){ float ph; return comb(bpm,&ph,0.25f); }
EXPORT void set_grid(float bpm,float offs){
  if(bpm<30) bpm=30; if(bpm>300) bpm=300; BPM=bpm; OFFS=offs; double tt=0,hu=0;
  for(int f=0;f<NF;f++){ float bp=(f/FPS-OFFS)*BPM/60; float fr=bp-ffloor(bp); fPulse[f]=(float)d_exp(-4*fr);
    tt+=(0.5+1.4*fRms[f]+2.2*fPulse[f]*fBass[f])/FPS; fTT[f]=(float)tt; hu+=(0.004+0.01*fHigh[f])/3; fHue[f]=(float)hu; }
  reset();
}
EXPORT int analyze(const float *pcm,int n,float sr){
  if(!inited) init_tables();
  if(!fft_ok){ for(int i=0;i<1024;i++){ twr[i]=(float)d_sin(PI/2-2*PI*i/2048); twi[i]=(float)-d_sin(2*PI*i/2048); }
    for(int i=0;i<2048;i++) win[i]=(float)(0.5-0.5*d_sin(PI/2-2*PI*i/2048)); fft_ok=1; }
  float hop=sr/FPS; NF=(int)(n/hop); if(NF>MAXF) NF=MAXF; if(NF<1){ NF=0; return 0; }
  float bhz=sr/2048; int edges[25]; for(int k=0;k<=24;k++){ int e=(int)(40*d_exp(d_log(sr*0.5/40*0.99)*k/24)/bhz); if(e<1) e=1; if(e>1023) e=1023; if(k&&e<=edges[k-1]) e=edges[k-1]+1; edges[k]=e; }
  int b0=(int)(20/bhz)+1,b1=(int)(150/bhz),b2=(int)(600/bhz),b3=(int)(3000/bhz),b4=(int)(4000/bhz); if(b1<=b0) b1=b0+1;
  memset(pmag,0,sizeof(pmag));
  for(int f=0;f<NF;f++){ int c=(int)(f*hop)-1024; double e=0;
    for(int i=0;i<2048;i++){ int j=c+i; float v=(j>=0&&j<n)?pcm[j]:0; fre[i]=v*win[i]; fim[i]=0; e+=v*v; }
    fft(); float sb=0,sm=0,sh=0,kk=0,sn=0,ht=0,tot=0;
    for(int i=1;i<1024;i++){ float m=fsqrt(fre[i]*fre[i]+fim[i]*fim[i]); float lm=flog(1+m), d=lm-pmag[i]; pmag[i]=lm; if(d<0) d=0;
      fre[i]=m; if(i>=b0&&i<=b1){ sb+=m; kk+=d; } else if(i<=b3){ if(i>b2) sm+=m; sn+=d; } else { sh+=m; if(i>=b4) ht+=d; } tot+=d; }
    fBass[f]=sb/(b1-b0+1); fMid[f]=sm/(b3-b2); fHigh[f]=sh/(1023-b3); fRms[f]=fsqrt((float)(e/2048)); fKick[f]=kk; fSnare[f]=sn; fHat[f]=ht; fOn[f]=tot;
    for(int k=0;k<24;k++){ float s=0; for(int i=edges[k];i<edges[k+1];i++) s+=fre[i]*fre[i]; fMel[f*24+k]=flog(1+s/(edges[k+1]-edges[k])); } }
  norm(fBass,NF,.98f); norm(fMid,NF,.98f); norm(fHigh,NF,.98f); norm(fRms,NF,.98f); norm(fKick,NF,.98f); norm(fSnare,NF,.98f); norm(fHat,NF,.98f);
  { float p=pctl(fMel,NF*24,7,.99f); for(int i=0;i<NF*24;i++){ float v=fMel[i]/p; fMel[i]=v>1.3f?1.3f:v; } }
  envf(fBass,NF,.6f,.12f); envf(fMid,NF,.6f,.12f); envf(fHigh,NF,.6f,.12f); envf(fRms,NF,.3f,.05f);
  // onset envelope: remove local mean
  for(int f=0;f<NF;f++){ float m=0; int c=0; for(int j=f-15;j<=f+15;j++) if(j>=0&&j<NF){ m+=fOn[j]; c++; } scratch[f]=fmax_(0,fOn[f]-m/c); }
  for(int f=0;f<NF;f++) fOn[f]=scratch[f];
  float best=-1,bb=120,bph=0,ph;
  for(float bpm=70;bpm<180;bpm+=0.25f){ float s=comb(bpm,&ph,0.5f); float l=(float)(d_log(bpm/118.0)/0.6931/0.8); s*=(float)d_exp(-0.5*l*l); if(s>best){ best=s; bb=bpm; } }
  best=-1; float c0=bb; for(float bpm=c0-0.4f;bpm<c0+0.4f;bpm+=0.01f){ float s=comb(bpm,&ph,0.25f); if(s>best){ best=s; bb=bpm; bph=ph; } }
  { float r=ffloor(bb+0.5f); if(fabs_(bb-r)<0.06f){ bb=r; comb(bb,&bph,0.25f); } }
  set_grid(bb,bph/FPS); return NF;
}
EXPORT int synth(int nf,float bpm){   // no-audio demo features
  if(!inited) init_tables(); if(nf>MAXF) nf=MAXF; NF=nf;
  for(int f=0;f<NF;f++){ float bp=f/FPS*bpm/60; int bi=(int)bp; float fr=bp-bi, e=(float)d_exp(-5*fr), h=bp*2-(int)(bp*2);
    fBass[f]=0.9f*e; fKick[f]=fr<0.06f?1:0; fSnare[f]=(bi&1)&&fr<0.06f?1.2f:0; fHat[f]=h<0.12f?0.9f:0; fRms[f]=0.45f+0.3f*e;
    fMid[f]=0.45f+0.3f*fsin(f*0.013f); fHigh[f]=0.3f+0.25f*fsin(f*0.007f+1); fOn[f]=fKick[f];
    for(int k=0;k<24;k++) fMel[f*24+k]=clamp01(0.35f+0.5f*fsin(f*0.05f+k*0.6f)*e+0.25f*e*(k<5)); }
  set_grid(bpm,0); return NF;
}

// ------------------------------------------------------------------ scenes
static void sc_tunnel(float *d,float T){
  for(int p=0;p<NP;p++,d+=3){ float r=GRa[p], u=0.35f/r+T*0.8f, v=GTa[p]*0.95493f+0.3f*fsin(u*1.5f+T)+T*0.1f;
    float pp=fsin(6.2832f*u)*fcos(6.2832f*v); const float *c=lut(u*0.22f+0.16f*pp+aHue);
    float m=fsin(25.133f*u)>0?1:0.55f, cr=r*2; if(cr>1) cr=1; m*=cr*fsqrt(fsqrt(cr));
    float gl=1-r*r*8; gl=gl>0?gl*gl*aP*aB:0; d[0]=c[0]*m+gl; d[1]=c[1]*m+gl; d[2]=c[2]*m+gl; } }
static void sc_weave(float *d,float T){
  float zoom=13+6*fsin(T*0.21f), a=0.25f*fsin(T*0.07f), ca=fcos(a), sa=fsin(a); const u8 *C=CA[((aVar%3)+3)%3]; int sc=(int)(T*3);
  for(int p=0;p<NP;p++,d+=3){ float X=GXa[p],Y=GYa[p], gx=(X*ca-Y*sa)*zoom+T*1.5f, gy=(X*sa+Y*ca)*zoom+T*0.4f;
    float fx=ffloor(gx), fy=ffloor(gy); int ix=(int)fx, iy=(int)fy; fx=gx-fx; fy=gy-fy;
    float sx=fsin(3.14159f*fx), sy=fsin(3.14159f*fy);
    if(C[((iy+sc)&511)*512+(ix&511)]){ const float *c=lut(ix*0.013f+aHue); float m=(0.35f+aMel[((ix%24)+24)%24])*(0.25f+0.75f*sx)*(0.6f+0.4f*sy);
      d[0]=c[0]*m; d[1]=c[1]*m; d[2]=c[2]*m; }
    else { const float *c=lut(iy*0.021f+aHue+0.5f); float m=0.8f*(0.25f+0.75f*sy)*(0.6f+0.4f*sx); d[0]=c[0]*m; d[1]=c[1]*m; d[2]=c[2]*m; } } }

static inline void stamp(float *d,float x,float y,float th,const float *c,float br){
  int r=(int)(th*0.5f+1.5f), xi=(int)x, yi=(int)y;
  for(int j=-r;j<=r;j++){ int py=yi+j; if(py<0||py>=H) continue; for(int i=-r;i<=r;i++){ int px=xi+i; if(px<0||px>=W) continue;
      float dx=px+0.5f-x, dy=py+0.5f-y, a=th*0.5f+0.5f-fsqrt(dx*dx+dy*dy); if(a<=0) continue; if(a>1) a=1; a*=br;
      float *q=d+(py*W+px)*3; for(int k=0;k<3;k++){ float v=c[k]*a; if(v>q[k]) q[k]=v; } } } }
static void line(float *d,float x0,float y0,float x1,float y1,float th,const float *c,float br){
  float t0=0,t1=1,dx=x1-x0,dy=y1-y0; float pq[4][2]={{-dx,x0+4},{dx,W+4-x0},{-dy,y0+4},{dy,H+4-y0}};
  for(int i=0;i<4;i++){ float p=pq[i][0],q=pq[i][1]; if(p==0){ if(q<0) return; } else { float r=q/p; if(p<0){ if(r>t1) return; if(r>t0) t0=r; } else { if(r<t0) return; if(r<t1) t1=r; } } }
  float ax=x0+dx*t0, ay=y0+dy*t0, bx=x0+dx*t1, by=y0+dy*t1; dx=bx-ax; dy=by-ay; float len=fsqrt(dx*dx+dy*dy); int n=(int)len+1; dx/=n; dy/=n;
  for(int i=0;i<=n;i++) stamp(d,ax+dx*i,ay+dy*i,th,c,br); }
static void ring(float *d,float cx,float cy,float r,float th,const float *c,float br){
  if(cx+r<-4||cx-r>W+4||cy+r<-4||cy-r>H+4) return; int n=(int)(6.2832f*r)+8; if(n>4000) n=4000;
  for(int i=0;i<n;i++){ float a=6.2832f*i/n; float x=cx+r*fcos(a), y=cy+r*fsin(a); if(x>-3&&x<W+3&&y>-3&&y<H+3) stamp(d,x,y,th,c,br); } }
static void disk(float *d,float cx,float cy,float r,const float *c,float br){
  int x0=(int)(cx-r-1),x1=(int)(cx+r+2),y0=(int)(cy-r-1),y1=(int)(cy+r+2); if(x0<0) x0=0; if(y0<0) y0=0; if(x1>W) x1=W; if(y1>H) y1=H;
  for(int y=y0;y<y1;y++) for(int x=x0;x<x1;x++){ float dx=x+0.5f-cx, dy=y+0.5f-cy, a=r+0.5f-fsqrt(dx*dx+dy*dy); if(a<=0) continue; if(a>1) a=1;
      float *q=d+(y*W+x)*3; for(int k=0;k<3;k++) q[k]=q[k]*(1-a)+c[k]*a*br; } }
static const float ZP[10][2]={{-.25f,.75f},{-.55f,0},{-.3f,.35f},{-.25f,-.75f},{-.3f,-.35f},{.3f,-.35f},{.25f,-.75f},{.3f,.35f},{.55f,0},{.25f,.75f}};
static const int SYZ[5][2]={{0,9},{1,8},{2,7},{3,6},{4,5}}, GATES[6][2]={{2,3},{4,1},{5,6},{7,1},{8,9},{6,3}}, LOOP[6]={1,2,7,8,5,4};
static void sc_numo(float *d,float T){
  static const float white[3]={1,1,1}, black[3]={0,0,0};
  const float *c0=PLINE[PALi][0], *c1=PLINE[PALi][1], *c2=PLINE[PALi][2];
  float ph=T*0.13f; ph-=ffloor(ph); int tw=aVar%4;
  for(int p=0;p<NP;p++){ float r=GRa[p]; float rg=fsin(flog(r)*14-T*1.5f+GTa[p]*tw)>0.96f?1:0; d[p*3]=rg*(0.10f*aP*c1[0]+0.18f*c0[0]); d[p*3+1]=rg*(0.10f*aP*c1[1]+0.18f*c0[1]); d[p*3+2]=rg*(0.10f*aP*c1[2]+0.18f*c0[2]); }
  for(int k=0;k<5;k++){ float p=(ph+k)/5, s=0.12f*(float)d_exp(p*5.5*0.6931), b=fsin(3.14159f*p); b=b*fsqrt(b);
    float ang=T*0.11f*((k&1)?1:-1)+k, c=fcos(ang), sn=fsin(ang), hs=H*0.5f; float px[10],py[10];
    for(int z=0;z<10;z++){ px[z]=(ZP[z][0]*c-ZP[z][1]*sn)*s*hs+W*0.5f; py[z]=(ZP[z][0]*sn+ZP[z][1]*c)*s*hs+H*0.5f; }
    float th=(s<1.2f?1.2f:2.2f)*SS;
    for(int i=0;i<6;i++){ int a=LOOP[i],e=LOOP[(i+1)%6]; line(d,px[a],py[a],px[e],py[e],th,c0,b); }
    for(int i=0;i<6;i++) line(d,px[GATES[i][0]],py[GATES[i][0]],px[GATES[i][1]],py[GATES[i][1]],1.0f*SS,c2,b*0.8f);
    for(int i=0;i<5;i++) line(d,px[SYZ[i][0]],py[SYZ[i][0]],px[SYZ[i][1]],py[SYZ[i][1]],th,c1,b);
    for(int z=0;z<10;z++){ float e=aMel[(z*2+3)%24], rad=s*hs*(0.05f+0.07f*e);
      disk(d,px[z],py[z],rad,black,1); if(e>0.75f) disk(d,px[z],py[z],rad,white,b); else ring(d,px[z],py[z],rad,th,c0,b); } } }
static void sc_flesh(float *d,float T){
  float cx[6],cy[6],w[6]; for(int k=0;k<6;k++){ cx[k]=1.2f*fsin(T*(0.23f+0.07f*k)+k*2.1f); cy[k]=0.7f*fcos(T*(0.19f+0.05f*k)+k*1.3f); w[k]=0.05f+0.03f*k+0.06f*aB; }
  for(int p=0;p<NP;p++,d+=3){ float X=GXa[p],Y=GYa[p],f=0; for(int k=0;k<6;k++){ float dx=X-cx[k],dy=Y-cy[k]; f+=w[k]/(dx*dx+dy*dy+0.01f); }
    float v=flog(1+f)*0.33f-T*0.07f+aHue*0.3f; const float *c=lut(v); float m=fsin(v*37.7f)>0?1:0.7f; d[0]=c[0]*m; d[1]=c[1]*m; d[2]=c[2]*m; } }
static void sc_rotor(float *d,float T){
  float a=T*1.1f, c=fcos(a), s=fsin(a), fq=5+9*aM; const float *c1=lut(aHue*2+0.13f), *c2=lut(aHue*2+0.55f);
  float rr=T*0.5f; rr=0.3f+0.9f*(rr-ffloor(rr));
  for(int p=0;p<NP;p++,d+=3){ float X=GXa[p],Y=GYa[p], dd=X*c+Y*s, e=-X*s+Y*c, st=fsin(dd*fq)*fsin(e*2.5f+T*2);
    if(st>0){ d[0]=c1[0]; d[1]=c1[1]; d[2]=c1[2]; } else { d[0]=c2[0]*0.9f; d[1]=c2[1]*0.9f; d[2]=c2[2]*0.9f; }
    if(fabs_(GRa[p]-rr)<0.04f){ d[0]=1-d[0]; d[1]=1-d[1]; d[2]=1-d[2]; } } }
static void sc_moire(float *d,float T){
  float cx=0.6f*fsin(T*0.31f), cy=0.35f*fcos(T*0.43f), fq=38+22*aM;
  for(int p=0;p<NP;p++,d+=3){ float X=GXa[p],Y=GYa[p]; float r1=fsqrt((X-cx)*(X-cx)+(Y-cy)*(Y-cy)), r2=fsqrt((X+cx)*(X+cx)+(Y+cy)*(Y+cy));
    float v=(fsin(r1*fq-T*3)+fsin(r2*fq+T*3))*0.25f+0.5f; const float *c=lut(v*0.5f+aHue); float m=v>0.5f?1:0.4f; d[0]=c[0]*m; d[1]=c[1]*m; d[2]=c[2]*m; } }
static void sc_spectro(float *d,float T){
  int radial=aVar&1; float asp=(float)W/H;
  for(int p=0;p<NP;p++,d+=3){ float bf; int age;
    if(radial){ bf=fabs_(GTa[p])*7.3211f; age=(int)(GRa[p]*70); } else { bf=fabs_(GXa[p])/asp*23; age=(int)((GYa[p]+1)*60); }
    int f=aF-age; if(f<0) f=0; int b0=(int)bf; if(b0>22) b0=22; float fr=bf-b0; if(fr>1) fr=1;
    const float *m=fMel+f*24; float v=m[b0]*(1-fr)+m[b0+1]*fr; const float *c=lut(v*0.45f+aHue+age*0.004f); float g=v*v*1.4f; if(g>1) g=1;
    d[0]=c[0]*g; d[1]=c[1]*g; d[2]=c[2]*g; } (void)T; }
static void sc_cells(float *d,float T){
  float cx[14],cy[14]; for(int k=0;k<14;k++){ cx[k]=1.6f*fsin(T*(0.11f+0.03f*k)+k*1.7f); cy[k]=0.9f*fcos(T*(0.13f+0.02f*k)+k*2.9f); }
  float core=0.05f*(1+2*aB*aP);
  for(int p=0;p<NP;p++,d+=3){ float X=GXa[p],Y=GYa[p],d1=1e9f,d2=1e9f; int i1=0;
    for(int k=0;k<14;k++){ float dx=X-cx[k],dy=Y-cy[k],q=dx*dx+dy*dy; if(q<d1){ d2=d1; d1=q; i1=k; } else if(q<d2) d2=q; }
    d1=fsqrt(d1); d2=fsqrt(d2); const float *c=lut(i1*0.071f+d1*0.9f-T*0.05f+aHue);
    float m=(d2-d1<0.03f)?0.08f:(fsin(d1*40-T*4)>0?1:0.6f); if(d1<core){ d[0]=d[1]=d[2]=1; } else { d[0]=c[0]*m; d[1]=c[1]*m; d[2]=c[2]*m; } } }

typedef void (*scene_fn)(float*,float);
static const scene_fn SCN[8]={sc_tunnel,sc_weave,sc_numo,sc_flesh,sc_rotor,sc_moire,sc_spectro,sc_cells};
static const int DEFPAL[16]={1,3,2,4,1,2,6,3, 3,1,7,4,2,2,1,6};
static void draw(const Blk *b,float *dst,int f){
  int sc=(int)b->scene; if(sc<0||sc>47) sc=0; int pal=(int)b->pal; if(pal<=0||pal>=NPAL) pal=sc<16?DEFPAL[sc]:1; sc&=7;   /* CPU path: only the first 8 scenes */
  PALi=pal; LUTp=LUT[pal]; aVar=(int)b->var; aF=f; aB=fBass[f]; aM=fMid[f]; aH=fHigh[f]; aP=fPulse[f]; aHue=fHue[f]+aVar*0.137f; aMel=fMel+f*24;
  SCN[sc](dst,fTT[f]*b->speed+aVar*7.3f);
  for(int i=0;i<NP*3;i++){ float v=dst[i]; dst[i]=v<0?0:(v>1?1:v); }
}

// ------------------------------------------------------------------ datamosh core
static void make_gray(const float *s,u8 *g){ int st=W/GW; if(st<1) st=1; int h2=st/2;
  for(int y=0;y<GH;y++) for(int x=0;x<GW;x++){ const float *a=s+((y*st)*W+x*st)*3, *b=s+((y*st+h2)*W+x*st+h2)*3;
      float l=(0.3f*(a[0]+b[0])+0.59f*(a[1]+b[1])+0.11f*(a[2]+b[2]))*127.5f; g[y*GW+x]=(u8)(l>255?255:l); } }
static void motion(float gain){ float sc=(float)W/GW*gain, lim=56.f*SS; int pdx=0,pdy=0;
  for(int by=0;by<BH;by++) for(int bx=0;bx<BW;bx++){ int x0=bx*8,y0=by*8,bh=(y0+8>GH)?GH-y0:8; int best=1<<30,bdx=0,bdy=0; if(!bx){ pdx=0; pdy=0; }
    for(int pass=0;pass<2;pass++) for(int dy=-6;dy<=6;dy++){ if(y0+dy<0||y0+dy+bh>GH) continue;
      for(int dx=-6;dx<=6;dx++){ if(pass==0&&(dx||dy)) continue; if(pass==1&&!dx&&!dy) continue; if(x0+dx<0||x0+dx+8>GW) continue;
        int ex=dx-pdx, ey=dy-pdy; int cost=2*((dx<0?-dx:dx)+(dy<0?-dy:dy))+2*((ex<0?-ex:ex)+(ey<0?-ey:ey));
        for(int y=0;y<bh&&cost<best;y++){ const u8 *c=G+(y0+y)*GW+x0, *q=PG+(y0+y+dy)*GW+x0+dx; for(int x=0;x<8;x++){ int e=c[x]-q[x]; cost+=e<0?-e:e; } }
        if(cost<best){ best=cost; bdx=dx; bdy=dy; } }
      if(pass==0&&best<=2*((pdx<0?-pdx:pdx)+(pdy<0?-pdy:pdy))) break; }
    pdx=bdx; pdy=bdy;
    float vx=bdx*sc, vy=bdy*sc; if(vx>lim) vx=lim; if(vx<-lim) vx=-lim; if(vy>lim) vy=lim; if(vy<-lim) vy=-lim; mvx[by*BW+bx]=vx; mvy[by*BW+bx]=vy; } }
static void warp(int coarse){ int bs=16*SS;
  for(int y=0;y<H;y++){ int by=y/bs; if(by>=BH) by=BH-1; if(coarse) by&=~1;
    for(int x=0;x<W;x++){ int bx=x/bs; if(bx>=BW) bx=BW-1; if(coarse) bx&=~1; int i=by*BW+bx; int sx=x+(int)mvx[i], sy=y+(int)mvy[i];
      if(sx<0) sx=-sx; if(sx>=W) sx=2*W-2-sx; if(sy<0) sy=-sy; if(sy>=H) sy=2*H-2-sy;
      const float *q=B+(sy*W+sx)*3; float *t=Tm+(y*W+x)*3; t[0]=q[0]; t[1]=q[1]; t[2]=q[2]; } }
  float *t=B; B=Tm; Tm=t; }
static void corrupt(const Blk *b,int f){ float g=b->glitch; if(g<=0) return; int bs=16*SS; rs=(u32)f*2654435761u+977u; rnd();
  if(fSnare[f]>1.15f-0.5f*g){ int n=2+rnd()%4; for(int k=0;k<n;k++){ int y0=(rnd()%(H/bs))*bs, hh=(1+rnd()%4)*bs, sh=((int)(rnd()%25)-12)*bs; if(y0+hh>H) hh=H-y0; if(sh<0) sh+=W;
      for(int y=y0;y<y0+hh;y++){ float *row=B+y*W*3; memcpy(Tm,row,sizeof(float)*W*3); memcpy(row+sh*3,Tm,sizeof(float)*(W-sh)*3); memcpy(row,Tm+(W-sh)*3,sizeof(float)*sh*3); } } }
  if(fHat[f]>1.1f-0.5f*g){ int n=(int)((4+14*fHigh[f])*g*1.5f);
    for(int k=0;k<n;k++){ int s=bs*(1+rnd()%4), y0=(rnd()%((H-s)/bs))*bs, x0=(rnd()%((W-s)/bs))*bs;
      if(rnd()&1){ float m[3]={0,0,0}; for(int y=0;y<s;y+=4) for(int x=0;x<s;x+=4){ const float *q=B+((y0+y)*W+x0+x)*3; m[0]+=q[0]; m[1]+=q[1]; m[2]+=q[2]; }
        float inv=16.f/(s*s); m[0]*=inv; m[1]*=inv; m[2]*=inv; for(int y=0;y<s;y++){ float *q=B+((y0+y)*W+x0)*3; for(int x=0;x<s;x++,q+=3){ q[0]=m[0]; q[1]=m[1]; q[2]=m[2]; } } }
      else { int y1=(rnd()%((H-s)/bs))*bs, x1=(rnd()%((W-s)/bs))*bs, sw=(rnd()%10)<3; if(y1==y0&&x1==x0) continue;
        for(int y=0;y<s;y++){ float *q=B+((y0+y)*W+x0)*3; const float *r=B+((y1+y)*W+x1)*3; for(int x=0;x<s;x++,q+=3,r+=3){ if(sw){ float t0=r[2],t2=r[0]; q[1]=r[1]; q[0]=t0; q[2]=t2; } else { q[0]=r[0]; q[1]=r[1]; q[2]=r[2]; } } } } } } }

static void emit(int f,int sh,int inv,float fade){ rs=(u32)f*747796405u+1u; u8 *o=OUT;
  for(int y=0;y<H;y++){ const float *row=B+y*W*3; for(int x=0;x<W;x++,o+=4){ int xr=x-sh, xb=x+sh; if(xr<0) xr+=W; if(xb>=W) xb-=W;
      float r=clamp01(row[xr*3]), g=clamp01(row[x*3+1]), b=clamp01(row[xb*3+2]);
      if(inv==1){ r=1-r; g=1-g; b=1-b; } else if(inv==2){ r*=1.6f; g*=0.24f; b*=0.16f; }
      rs=rs*1664525u+1013904223u; float n=((int)(rs>>16&255)-128)*(0.055f/128), v=VIG[y*W+x]*fade;
      r=(r*v+n*fade)*255; g=(g*v+n*fade)*255; b=(b*v+n*fade)*255;
      o[0]=(u8)(r<0?0:(r>255?255:r)); o[1]=(u8)(g<0?0:(g>255?255:g)); o[2]=(u8)(b<0?0:(b>255?255:b)); o[3]=255; } } }

enum { T_CUT=0, T_XFADE=1, T_MOSH=2, T_MELT=3, T_STROBE=4 };
// ---- per-frame decisions, shared by the CPU renderer and the WebGPU plan ----
typedef struct { int black,iframe,skipres,freeze,inv,coarse; const Blk *a,*b,*cur; float xf,leak,gain,fade,shf,bp; } Plan;
static Plan P;
static Blk EB[3];
static float modsrc(int s,int f,float bp){ switch(s){ case 1:return fBass[f]; case 2:return fMid[f]; case 3:return fHigh[f]; case 4:return fRms[f]; case 5:return fPulse[f];
  case 6:return 0.5f+0.5f*fsin(6.2832f*bp); case 7:return 0.5f+0.5f*fsin(1.5708f*bp); case 8:return 0.5f+0.5f*fsin(0.3927f*bp);
  case 9:{ float x=bp*0.25f; return x-ffloor(x); } case 10:return fKick[f]>1?1:fKick[f]; default:return 0; } }
static const Blk *eff(const Blk *b,int f,float bp,int slot){ if(!b) return 0; Blk *e=&EB[slot]; *e=*b; e->hueo=0; const float *m=&b->ms0;
  for(int i=0;i<3;i++){ int s=(int)m[i*3]; if(s<=0) continue; float v=modsrc(s,f,bp<0?0:bp)*m[i*3+2];
    switch((int)m[i*3+1]){ case 0:e->dens=clamp01(e->dens+v); break; case 1:e->mosh=clamp01(e->mosh+v); break; case 2:e->glitch=clamp01(e->glitch+v); break; case 3:e->rgb=fmax_(0,e->rgb+v*2); break; case 4:e->hueo=v; break; } }
  return e; }
static void decide(int f){
  if(f<lastFlashF) lastFlashF=-100000;   /* a backwards jump without reset() must not suppress flashes */
  float bp=(f/FPS-OFFS)*BPM/60; int k=-1; for(int i=0;i<NB;i++) if(bp>=BL[i].start&&bp<BL[i].start+BL[i].len){ k=i; break; }
  P.bp=bp; P.fade=fmin_(1,f/12.f)*clamp01((NF-1-f)/60.f); P.black=0; P.a=P.b=P.cur=0; P.xf=-1; P.inv=0; P.skipres=P.freeze=P.coarse=0;
  if(k<0){ P.black=1; lastK=-1; lastF=f; needI=1; return; }
  const Blk *b=&BL[k]; int entered=(k!=lastK), iframe=needI, flash=0, tr=(int)b->trans; needI=0;
  if(f<lastF||f>lastF+8) iframe=1;
  if(entered){ if(lastK<0||tr==T_CUT) iframe=1; else if(tr==T_STROBE){ iframe=1; flash=1; } else if(tr!=T_XFADE) P.skipres=1; }
  const Blk *use=b; float into=bp-b->start;
  if(tr==T_XFADE&&k>0&&b->tlen>0&&into<b->tlen&&fabs_(BL[k-1].start+BL[k-1].len-b->start)<0.02f){ float x=into/b->tlen; P.xf=x*x*(3-2*x); }
  int sel=0;
  if(k+1<NB&&fabs_(BL[k+1].start-(b->start+b->len))<0.02f){ const Blk *nx=&BL[k+1]; float dd=b->start+b->len-bp; int nt=(int)nx->trans;
    if(nt==T_MELT&&dd<nx->tlen) P.freeze=1;
    if(nt==T_STROBE&&dd<nx->tlen){ float fpb=FPS*60/BPM, stp=0.5f;
      if(nx->strobe<3){ while(stp*fpb<12&&stp<4) stp*=2; }   /* no full-frame change faster than 12 frames unless the block opts in */
      sel=((int)(dd/stp))&1; if(sel) use=nx; } }
  if(sel!=lastSel){ iframe=1; flash=1; } lastSel=sel;
  if(P.xf>=0){ P.a=eff(&BL[k-1],f,bp,0); P.b=eff(b,f,bp,1); } else P.a=eff(use,f,bp,0);
  b=eff(b,f,bp,2); P.cur=b; float m=clamp01(b->mosh), q=1-m;
  if(b->bloom>0){ float fr=bp*0.25f; fr-=ffloor(fr); if(fr>1-b->bloom*0.5f) P.freeze=1; } P.leak=q*q*q*q*q; if(P.leak<0.004f) P.leak=0.004f; P.gain=1+2.2f*m; P.coarse=m>0.85f;
  if(m<=0.001f) iframe=1; P.iframe=iframe;
  P.shf=(2+16*fBass[f]*fPulse[f])*b->rgb;
  if(b->strobe>0){ int bi=(int)ffloor(bp); float since=(bp-bi)*FPS*60/BPM; float kk=fKick[f]; if(f>0) kk=fmax_(kk,fKick[f-1]); if(f+1<NF) kk=fmax_(kk,fKick[f+1]);
    if(since<2&&kk>0.35f&&(b->strobe>=2||!(bi&1))) P.inv=since<1?1:2; }
  if(flash) P.inv=1;
  if(P.inv){                                       /* photosensitivity: >=12 frames between flash starts, all sources */
    if(b->strobe>=3||f-lastFlashF>=12) lastFlashF=f;
    else if(!(P.inv==2&&f-lastFlashF==1)) P.inv=0; /* the tint frame belongs to the flash that just started */
  }
  lastK=k; lastF=f;
}
EXPORT void render(int f){                       /* CPU renderer */
  if(NF<1){ memset(OUT,0,NP*4); return; }
  if(f<0) f=0; if(f>=NF) f=NF-1;
  decide(f);
  if(P.black){ for(int i=0;i<NP*3;i++) B[i]*=0.88f; emit(f,0,0,P.fade); return; }
  if(P.xf>=0){ draw(P.a,Tm,f); draw(P.b,S,f); float a=P.xf; for(int i=0;i<NP*3;i++) S[i]=Tm[i]*(1-a)+S[i]*a; } else draw(P.a,S,f);
  make_gray(S,G);
  if(P.iframe) memcpy(B,S,sizeof(float)*NP*3);
  else {
    if(!P.freeze) motion(P.gain); else for(int i=0;i<BW*BH;i++){ mvx[i]*=1.02f; mvy[i]*=1.02f; }
    warp(P.coarse);
    if(!P.skipres&&!P.freeze){ float leak=P.leak; for(int i=0;i<NP*3;i++){ float v=B[i]+S[i]-Pv[i]; v+=(S[i]-v)*leak; B[i]=v<-0.15f?-0.15f:(v>1.15f?1.15f:v); } }
  }
  { float *t=S; S=Pv; Pv=t; u8 *g=G; G=PG; PG=g; }
  corrupt(P.cur,f);
  emit(f,(int)(P.shf*SS),P.inv,P.fade);
}
// ---- WebGPU plan: everything the shaders need for frame f, as floats ----
#define PLN 512
static float PL[PLN];
static void pack_scene(float *o,const Blk *b,int f){
  int sc=(int)b->scene; if(sc<0||sc>47) sc=0; int pal=(int)b->pal; o[19]=(pal<=0||pal>=NPAL)?0:pal; if(pal<=0||pal>=NPAL) pal=sc<16?DEFPAL[sc]:1; int v=(int)b->var;
  o[0]=sc; o[1]=pal; o[2]=fTT[f]*b->speed+v*7.3f; o[3]=fHue[f]+v*0.137f+b->hueo; o[4]=v; o[5]=b->dens; o[6]=fBass[f]; o[7]=fMid[f]; o[8]=fHigh[f]; o[9]=fPulse[f];
  for(int j=0;j<3;j++) for(int c=0;c<3;c++) o[10+j*3+c]=PLINE[pal][j][c];
}
EXPORT float *plan(int f){
  memset(PL,0,sizeof(PL)); if(NF<1){ PL[0]=1; return PL; }
  if(f<0) f=0; if(f>=NF) f=NF-1;
  decide(f);
  PL[0]=P.black; PL[9]=P.fade; PL[11]=P.bp; if(P.black) return PL;
  PL[1]=P.iframe; PL[2]=P.skipres||P.freeze; PL[3]=P.freeze; PL[4]=P.leak; PL[5]=P.gain; PL[6]=P.coarse; PL[7]=P.shf/640.f; PL[8]=P.inv; PL[10]=P.xf;
  pack_scene(PL+16,P.a,f); if(P.b) pack_scene(PL+40,P.b,f);
  for(int k=0;k<24;k++) PL[64+k]=fMel[f*24+k];
  // glitch ops: type(1 row shift,2 flat fill,3 copy,4 copy+channel swap), rect x,y,w,h, source offset x,y  (all in 0..1 units)
  const Blk *b=P.cur; float g=b->glitch; int n=0; float *o=PL+96; const float bx=1.f/40, by=16.f/360;
  if(g>0){ rs=(u32)f*2654435761u+977u; rnd();
    if(fSnare[f]>1.15f-0.5f*g){ int c=2+rnd()%4; for(int k=0;k<c&&n<48;k++,n++){ float y0=(rnd()%22)*by, hh=(1+rnd()%4)*by, sh=((int)(rnd()%25)-12)*bx;
        o[n*8]=1; o[n*8+1]=0; o[n*8+2]=y0; o[n*8+3]=1; o[n*8+4]=hh; o[n*8+5]=-sh; o[n*8+6]=0; } }
    if(fHat[f]>1.1f-0.5f*g){ int c=(int)((4+14*fHigh[f])*g*1.5f); for(int k=0;k<c&&n<48;k++,n++){ int s=1+rnd()%4, y0=rnd()%(22-s), x0=rnd()%(40-s);
        if(rnd()&1){ o[n*8]=2; o[n*8+5]=0; o[n*8+6]=0; } else { int y1=rnd()%(22-s), x1=rnd()%(40-s); o[n*8]=(rnd()%10)<3?4:3; o[n*8+5]=(x1-x0)*bx; o[n*8+6]=(y1-y0)*by; }
        o[n*8+1]=x0*bx; o[n*8+2]=y0*by; o[n*8+3]=s*bx; o[n*8+4]=s*by; } } }
  PL[12]=n;
  for(int i=0;i<NFX;i++){ const Fx *x=&FXB[i]; int t=(int)x->type; if(t<0||t>15) continue; if(P.bp>=x->start&&P.bp<x->start+x->len&&x->amt>PL[480+t]){ PL[480+t]=x->amt; PL[496+t]=x->par; } }
  return PL;
}
EXPORT float *thumb_plan(int i,int f){ memset(PL,0,sizeof(PL)); if(NF<1||i<0||i>=NB) return PL; if(f<0) f=0; if(f>=NF) f=NF-1;
  PL[11]=(f/FPS-OFFS)*BPM/60; pack_scene(PL+16,&BL[i],f); for(int k=0;k<24;k++) PL[64+k]=fMel[f*24+k]; return PL; }
EXPORT float *pal_ptr(void){ return (float*)LUT; }
EXPORT u8 *ca_ptr(void){ return (u8*)CA; }
EXPORT float *mel_ptr(void){ return fMel; }
