/* XMRig — batched AVX-512 RandomX: self-test / verify drivers. */
#include "crypto/batchedx/BatchedInternal.h"
#include "crypto/randomx/randomx.h"
#include "crypto/randomx/virtual_machine.hpp"
#include "crypto/randomx/blake2/blake2.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxVm.h"
#include "crypto/rx/RxConfig.h"
#include "crypto/common/Assembly.h"
#include "base/tools/Buffer.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace xmrig {
namespace batchedx {

// [default target] all std::vector / xmrig Rx work lives here, OUTSIDE the AVX-512
// pragma region, so the vector allocator is never instantiated under a target mismatch.
struct Stage8Ctx { randomx_vm* vm; const uint64_t* ds; uint8_t* scratch; };
static Stage8Ctx stage8_setup()
{
    static const char* key = "batchedx-stage8";
    static Buffer seed((const uint8_t*)key, (const uint8_t*)key + strlen(key));
    static RxDataset dataset(false, false, true, RxConfig::FastMode, 0);
    static bool inited = false;
    Stage8Ctx c{ nullptr, nullptr, nullptr };
    fprintf(stderr, "[8] ctor: dataset=%p cache=%p\n", (void*)dataset.get(), (void*)dataset.cache());
    if (!dataset.get()) { printf("  8: dataset alloc failed (FastMode needs ~2GiB)\n"); return c; }
    if (!inited) { fprintf(stderr, "[8] init dataset...\n"); bool ok = dataset.init(seed, 1, 0); fprintf(stderr, "[8] init=%d\n", ok); inited = true; }
    fprintf(stderr, "[8] create vm...\n");
    static uint8_t* refScratch = (uint8_t*)aligned_alloc(4096, (size_t)RandomX_CurrentConfig.ScratchpadL3_Size);
    c.vm = RxVm::create(&dataset, refScratch, false /*hardware AES; matches batched fillAes<false>*/, Assembly(), 0);
    c.scratch = refScratch;
    fprintf(stderr, "[8] vm=%p\n", (void*)c.vm);
    c.ds = (const uint64_t*)randomx_get_dataset_memory(dataset.get());
    fprintf(stderr, "[8] ds=%p\n", (void*)c.ds);
    return c;
}
static void stage8_teardown(randomx_vm* vm) { if (vm) RxVm::destroy(vm); }

#if defined(__GNUC__)
#   pragma GCC push_options
#   pragma GCC target("avx512f,avx512dq")
#endif

static uint64_t rng_s;
static inline uint64_t rng() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }

static inline bool bothNaN(uint64_t a, uint64_t b) {
    // exponent all ones + nonzero mantissa = NaN (ignore sign)
    auto isnan = [](uint64_t x){ return ((x & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL)
                                     && (x & 0x000fffffffffffffULL); };
    return isnan(a) && isnan(b);
}

int verifyIntegerOps(){
  printf("== BatchedVm Stage 1: integer ops verify ==\n");
  rng_s = 0x9e3779b97f4a7c15ULL;

  const int PROGRAMS = 20000;
  const int PROGLEN  = 256;
  int fails = 0;

  for (int p = 0; p < PROGRAMS; ++p) {
    // random program (integer register ops only)
    BInsn prog[PROGLEN] = {};
    for (int i = 0; i < PROGLEN; ++i) {
      uint64_t r = rng();
      prog[i].op  = (BOp)(r % 10);
      prog[i].dst   = (r >> 8)  & 7;
      prog[i].src   = (r >> 16) & 7;
      prog[i].shift = (r >> 24) & 3;
      prog[i].imm   = rng();
    }

    // random initial registers for 8 lanes
    uint64_t batched[IREGS][LANES];
    uint64_t scalar [LANES][IREGS];
    for (int lane = 0; lane < LANES; ++lane)
      for (int k = 0; k < IREGS; ++k) {
        uint64_t v = rng();
        scalar[lane][k]   = v;
        batched[k][lane]  = v;
      }

    // run both
    runIntegerProgram(batched, prog, PROGLEN);
    for (int lane = 0; lane < LANES; ++lane)
      scalarIntegerProgram(scalar[lane], prog, PROGLEN);

    // diff
    for (int lane = 0; lane < LANES; ++lane)
      for (int k = 0; k < IREGS; ++k)
        if (batched[k][lane] != scalar[lane][k]) {
          if (fails < 5)
            printf("  MISMATCH prog %d lane %d reg %d: batched %016llx scalar %016llx\n",
                 p, lane, k,
                 (unsigned long long)batched[k][lane],
                 (unsigned long long)scalar[lane][k]);
          ++fails;
        }
  }

  printf("== %s (%d mismatches over %d programs x %d lanes) ==\n",
       fails == 0 ? "ALL PASS" : "FAILURES", fails, PROGRAMS, LANES);
  return fails == 0 ? 0 : 1;
}

int verifyFloatOps()
{
    printf("== BatchedVm Stage 3: float ops verify ==\n");
    rng_s = 0xd1b54a32d192 ^ 0x9e3779b97f4a7c15ULL;

    const int PROGRAMS = 20000;
    const int PROGLEN  = 256;
    long fails = 0;

    for (int p = 0; p < PROGRAMS; ++p) {
        FInsn prog[PROGLEN];
        for (int i = 0; i < PROGLEN; ++i) {
            uint64_t r = rng();
            prog[i].op  = (FOp)(r % 6);
            prog[i].dst = (r >> 8)  & 3;
            prog[i].src = (r >> 16) & 3;
        }

        // random finite doubles for 8 lanes x FREGS x 2
        BFReg batched[FREGS];
        double scalar[LANES][FREGS][2];
        for (int k = 0; k < FREGS; ++k) {
            double lo[LANES], hi[LANES];
            for (int lane = 0; lane < LANES; ++lane) {
                // build a finite double in a modest range to avoid inf/nan divergence noise
                uint64_t r0 = rng(), r1 = rng();
                double v0 = 1.0 + (double)(r0 & 0xffffffff) / 4294967296.0 * 1e6;
                double v1 = 1.0 + (double)(r1 & 0xffffffff) / 4294967296.0 * 1e6;
                lo[lane] = v0; hi[lane] = v1;
                scalar[lane][k][0] = v0; scalar[lane][k][1] = v1;
            }
            batched[k].lo = _mm512_loadu_pd(lo);
            batched[k].hi = _mm512_loadu_pd(hi);
        }

        runFloatProgram(batched, prog, PROGLEN);
        for (int lane = 0; lane < LANES; ++lane)
            scalarFloatProgram(scalar[lane], prog, PROGLEN);

        // diff by raw bits (exact)
        for (int k = 0; k < FREGS; ++k) {
            double blo[LANES], bhi[LANES];
            _mm512_storeu_pd(blo, batched[k].lo);
            _mm512_storeu_pd(bhi, batched[k].hi);
            for (int lane = 0; lane < LANES; ++lane) {
                uint64_t xb, xs;
                memcpy(&xb, &blo[lane], 8); memcpy(&xs, &scalar[lane][k][0], 8);
                if (xb != xs && !bothNaN(xb, xs)) { if (fails < 5) printf("  MISMATCH prog %d lane %d reg %d.lo\n", p, lane, k); ++fails; }
                memcpy(&xb, &bhi[lane], 8); memcpy(&xs, &scalar[lane][k][1], 8);
                if (xb != xs && !bothNaN(xb, xs)) { if (fails < 5) printf("  MISMATCH prog %d lane %d reg %d.hi\n", p, lane, k); ++fails; }
            }
        }
    }

    printf("== %s (%ld mismatches over %d programs x %d lanes) ==\n",
           fails == 0 ? "ALL PASS" : "FAILURES", fails, PROGRAMS, LANES);
    return fails == 0 ? 0 : 1;
}

int verifyMemoryOps()
{
    printf("== BatchedVm Stage 4: memory ops verify ==\n");
    rng_s = 0x243f6a8885a308d3ULL ^ 0x9e3779b97f4a7c15ULL;

    const int PROGRAMS = 5000;      // fewer: each carries a scratchpad
    const int PROGLEN  = 256;
    // small scratchpad for the test (must be power-of-2 words). 64 KiB = 8192 words.
    const uint64_t spWords = 8192;
    const uint32_t spMaskBytes = (uint32_t)(spWords * 8 - 8); // 8-byte aligned mask
    long fails = 0;

    // batched: one contiguous block of LANES*spWords; scalar: LANES separate arrays
    static uint64_t spB[LANES * 8192];
    static uint64_t spS[LANES][8192];

    // memory op set (indices into a small table)
    static const BOp memops[7] = { B_IADD_M, B_ISUB_M, B_IMUL_M, B_IMULH_M, B_ISMULH_M, B_IXOR_M, B_ISTORE };

    for (int p = 0; p < PROGRAMS; ++p) {
        // random program of memory ops
        BInsn prog[PROGLEN] = {};
        for (int i = 0; i < PROGLEN; ++i) {
            uint64_t r = rng();
            prog[i].op      = memops[r % 7];
            prog[i].dst     = (r >> 8)  & 7;
            prog[i].src     = (r >> 16) & 7;
            prog[i].shift   = 0;
            prog[i].imm     = rng();
            prog[i].memMask = spMaskBytes;   // fixed mask for the test scratchpad
        }

        // identical initial registers + scratchpads
        uint64_t batched[IREGS][LANES];
        uint64_t scalar [LANES][IREGS];
        for (int lane = 0; lane < LANES; ++lane) {
            for (int k = 0; k < IREGS; ++k) { uint64_t v = rng(); scalar[lane][k] = v; batched[k][lane] = v; }
            for (uint64_t w = 0; w < spWords; ++w) { uint64_t v = rng(); spS[lane][w] = v; spB[lane*spWords + w] = v; }
        }

        runMemoryProgram(batched, prog, PROGLEN, spB, spWords);
        for (int lane = 0; lane < LANES; ++lane)
            scalarMemoryProgram(scalar[lane], prog, PROGLEN, spS[lane]);

        // diff registers
        for (int lane = 0; lane < LANES; ++lane)
            for (int k = 0; k < IREGS; ++k)
                if (batched[k][lane] != scalar[lane][k]) {
                    if (fails < 5) printf("  MISMATCH prog %d lane %d reg %d: b=%016llx s=%016llx\n",
                        p, lane, k, (unsigned long long)batched[k][lane], (unsigned long long)scalar[lane][k]);
                    ++fails;
                }
        // diff scratchpads (ISTORE writes)
        for (int lane = 0; lane < LANES; ++lane)
            for (uint64_t w = 0; w < spWords; ++w)
                if (spB[lane*spWords + w] != spS[lane][w]) {
                    if (fails < 5) printf("  MISMATCH prog %d lane %d sp[%llu]\n", p, lane, (unsigned long long)w);
                    ++fails;
                }
    }

    printf("== %s (%ld mismatches over %d programs x %d lanes) ==\n",
           fails == 0 ? "ALL PASS" : "FAILURES", fails, PROGRAMS, LANES);
    return fails == 0 ? 0 : 1;
}

int verifyBranchOps()
{
    printf("== BatchedVm Stage 5: branch (CBRANCH) verify ==\n");
    rng_s = 0xb7e151628aed2a6bULL ^ 0x9e3779b97f4a7c15ULL;

    const int PROGRAMS = 20000;
    const int PROGLEN  = 256;
    long fails = 0;

    for (int p = 0; p < PROGRAMS; ++p) {
        BInsn prog[PROGLEN] = {};
        for (int i = 0; i < PROGLEN; ++i) {
            uint64_t r = rng();
            // ~10% CBRANCH, rest integer register ops (0..9)
            if ((r % 10) == 0) {
                prog[i].op      = B_CBRANCH;
                prog[i].dst     = (r >> 8) & 7;
                prog[i].imm     = rng();
                // condition mask: a few bits high enough that "==0" is plausible but not
                // always taken (mirrors RandomX ConditionMask placement)
                prog[i].memMask = (uint32_t)(0xffU << 8);   // bits 8..15
                prog[i].target  = (int16_t)(i > 0 ? (rng() % i) : 0);  // backward target
                prog[i].src     = 0;
                prog[i].shift   = 0;
            } else {
                prog[i].op      = (BOp)(r % 10);
                prog[i].dst     = (r >> 8)  & 7;
                prog[i].src     = (r >> 16) & 7;
                prog[i].shift   = (r >> 24) & 3;
                prog[i].imm     = rng();
                prog[i].memMask = 0;
                prog[i].target  = 0;
            }
        }

        uint64_t batched[IREGS][LANES];
        uint64_t scalar [LANES][IREGS];
        for (int lane = 0; lane < LANES; ++lane)
            for (int k = 0; k < IREGS; ++k) { uint64_t v = rng(); scalar[lane][k] = v; batched[k][lane] = v; }

        runBranchProgram(batched, prog, PROGLEN);
        for (int lane = 0; lane < LANES; ++lane)
            scalarBranchProgram(scalar[lane], prog, PROGLEN);

        for (int lane = 0; lane < LANES; ++lane)
            for (int k = 0; k < IREGS; ++k)
                if (batched[k][lane] != scalar[lane][k]) {
                    if (fails < 5) printf("  MISMATCH prog %d lane %d reg %d: b=%016llx s=%016llx\n",
                        p, lane, k, (unsigned long long)batched[k][lane], (unsigned long long)scalar[lane][k]);
                    ++fails;
                }
    }

    printf("== %s (%ld mismatches over %d programs x %d lanes) ==\n",
           fails == 0 ? "ALL PASS" : "FAILURES", fails, PROGRAMS, LANES);
    return fails == 0 ? 0 : 1;
}

int verifyProgram()
{
    RandomX_CurrentConfig.Apply();
    printf("== BatchedVm Stage 6a: real-program int/branch verify ==\n");
    const int PROGRAMS = 2000;
    long fails = 0;
    rng_s = 0xcafef00dd15ea5e5ULL ^ 0x9e3779b97f4a7c15ULL;
    for (int p=0;p<PROGRAMS;++p) {
        // real, valid program: random 64-byte seed -> AES fill (as the VM does)
        alignas(16) uint8_t seed[64];
        for (int i=0;i<64;++i) seed[i]=(uint8_t)rng();
        randomx::Program program;
        fillAes4Rx4<false>(seed, 128 + RandomX_CurrentConfig.ProgramSize * 8, &program);

        randomx::NativeRegisterFile nreg;
        static randomx::InstructionByteCode bytecode[512];
        randomx::BytecodeMachine bm;

        bm.compileProgram(program, bytecode, nreg);

        // translate to BProg (int/branch ops; float/mem flagged not-ok = no-op here)
        BProg bp; bp.count = (int)RandomX_CurrentConfig.ProgramSize;
        for (int i=0;i<bp.count;++i) bp.ok[i] = translateInt(bytecode[i], nreg, bp.ins[i]);

        // run batched (8 lanes) + scalar (8 lanes)
        uint64_t batched[IREGS][LANES], scalar[LANES][IREGS];
        for (int lane=0;lane<LANES;++lane)
            for (int k=0;k<IREGS;++k){uint64_t v=rng();scalar[lane][k]=v;batched[k][lane]=v;}
        runProgramIntBranch(batched, bp);
        for (int lane=0;lane<LANES;++lane) scalarProgramIntBranch(scalar[lane], bp);

        for (int lane=0;lane<LANES;++lane)
            for (int k=0;k<IREGS;++k)
                if (batched[k][lane]!=scalar[lane][k]) {
                    if (fails<5) printf("  MISMATCH prog %d lane %d reg %d\n",p,lane,k);
                    ++fails;
                }
    }
    printf("== %s (%ld mismatches over %d programs x %d lanes) ==\n",
           fails==0?"ALL PASS":"FAILURES", fails, PROGRAMS, LANES);
    return fails==0?0:1;
}

static int verifyProgramFullImpl(bool cfround, const char* stage)
{
    _mm_setcsr(0x9FC0);
    RandomX_CurrentConfig.Apply();
    printf("== BatchedVm Stage %s: full real-program verify%s ==\n",
           stage, cfround ? " (per-lane CFROUND)" : " (vs executeBytecode)");
    const int PROGRAMS = 1000;
    // full RandomX scratchpad (2 MiB / lane) so the real memMask stays in bounds.
    const uint64_t spWords = 2097152ull / 8;              // 262144 words = 2 MiB
    long fails = 0;
    rng_s = 0x2545F4914F6CDD1DULL ^ 0x9e3779b97f4a7c15ULL;

    // heap-allocate: 8 x 2 MiB each (too large for static/stack)
    uint64_t* spB   = (uint64_t*)malloc((size_t)LANES * spWords * 8);
    uint64_t* spSnapFlat = (uint64_t*)malloc((size_t)LANES * spWords * 8);
    if (!spB || !spSnapFlat) { printf("  6b: scratchpad alloc failed\n"); free(spB); free(spSnapFlat); return 1; }
    auto SNAP = [&](int lane)->uint64_t* { return spSnapFlat + (size_t)lane*spWords; };

    for (int p=0;p<PROGRAMS;++p) {
        alignas(16) uint8_t seed[64]; for(int i=0;i<64;++i) seed[i]=(uint8_t)rng();
        randomx::Program program;
        fillAes4Rx4<false>(seed, 128 + RandomX_CurrentConfig.ProgramSize*8, &program);

        randomx::NativeRegisterFile nregT;
        static randomx::InstructionByteCode btT[512];
        randomx::BytecodeMachine bmT; bmT.compileProgram(program, btT, nregT);

        randomx::ProgramConfiguration cfg;
        {
            auto staticExp=[&](uint64_t e){ uint64_t x=0x300; x|=(e>>(64-4))<<4; x<<=52; return x; };
            auto floatMask=[&](uint64_t e){ return (e & ((1ULL<<22)-1)) | staticExp(e); };
            cfg.eMask[0]=floatMask(program.getEntropy(14));
            cfg.eMask[1]=floatMask(program.getEntropy(15));
            cfg.readReg0=cfg.readReg1=cfg.readReg2=cfg.readReg3=0;
        }

        FullProg fp; fp.count=(int)RandomX_CurrentConfig.ProgramSize;
        for (int i=0;i<fp.count;++i) translateFull(btT[i], nregT, fp.ins[i], cfround);

        // ---- pristine input state (snapshot) ----
        uint64_t rSnap[IREGS][LANES];
        double fLo[8][LANES], fHi[8][LANES], aLo[4][LANES], aHi[4][LANES];
        for (int lane=0; lane<LANES; ++lane) {
            for (int k=0;k<IREGS;++k) rSnap[k][lane]=rng();
            for (int k=0;k<8;++k){ fLo[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0;
                                   fHi[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0; }
            for (int k=0;k<4;++k){ aLo[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0;
                                   aHi[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0; }
            for (uint64_t w=0; w<spWords; ++w) SNAP(lane)[w]=rng();
        }

        // ---- batched run (copies of inputs) ----
        uint64_t rB[IREGS][LANES];
        for (int k=0;k<IREGS;++k) for(int l=0;l<LANES;++l) rB[k][l]=rSnap[k][l];
        for (int lane=0; lane<LANES; ++lane)
            for (uint64_t w=0; w<spWords; ++w) spB[lane*spWords+w]=SNAP(lane)[w];
        BFReg Fb[8], Ab[4];
        for (int k=0;k<8;++k){ Fb[k].lo=_mm512_loadu_pd(fLo[k]); Fb[k].hi=_mm512_loadu_pd(fHi[k]); }
        for (int k=0;k<4;++k){ Ab[k].lo=_mm512_loadu_pd(aLo[k]); Ab[k].hi=_mm512_loadu_pd(aHi[k]); }
        runProgramFull(rB, Fb, Ab, spB, spWords, cfg.eMask, fp);
        // read back batched float f/e registers
        double Fblo[8][LANES], Fbhi[8][LANES];
        for (int k=0;k<8;++k){ _mm512_storeu_pd(Fblo[k],Fb[k].lo); _mm512_storeu_pd(Fbhi[k],Fb[k].hi); }

        // ---- scalar ground truth per lane via executeBytecode ----
        for (int lane=0; lane<LANES; ++lane) {
            randomx::NativeRegisterFile nr;
            static randomx::InstructionByteCode bt[512];
            randomx::BytecodeMachine bm; bm.compileProgram(program, bt, nr);
            if (!cfround)                                      // 6b neutralises CFROUND on both sides; 6c runs it for real
                for (int i=0;i<fp.count;++i)
                    if (bt[i].type==randomx::InstructionType::CFROUND) bt[i].type=randomx::InstructionType::NOP;
            for (int k=0;k<IREGS;++k) nr.r[k]=rSnap[k][lane];
            for (int k=0;k<randomx::RegisterCountFlt;++k){
                double lo=fLo[k][lane], hi=fHi[k][lane];
                nr.f[k]=_mm_set_pd(hi,lo);
                double elo=fLo[k+4][lane], ehi=fHi[k+4][lane];
                nr.e[k]=_mm_set_pd(ehi,elo);
                nr.a[k]=_mm_set_pd(aHi[k][lane],aLo[k][lane]);
            }
            uint8_t* sp = reinterpret_cast<uint8_t*>(SNAP(lane));
            rx_reset_float_state();                            // no rounding-mode leak lane->lane / prog->prog
            randomx::BytecodeMachine::executeBytecode(bt, sp, cfg);

            // diff integer regs
            for (int k=0;k<IREGS;++k)
                if (rB[k][lane]!=nr.r[k]) { if(fails<6) printf("  MISMATCH prog %d lane %d R%d\n",p,lane,k); ++fails; }
            // diff float f/e regs (raw bits, NaN-equal)
            auto cmpF=[&](int gidx, double blo, double bhi, rx_vec_f128 sv){
                alignas(16) double s[2]; _mm_store_pd(s, sv);
                uint64_t xb,xs;
                memcpy(&xb,&blo,8); memcpy(&xs,&s[0],8);
                bool n1=((xb&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)&&(xb&0xfffffffffffffULL);
                bool n2=((xs&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)&&(xs&0xfffffffffffffULL);
                if (xb!=xs && !(n1&&n2)) { if(fails<6) printf("  MISMATCH prog %d lane %d F%d.lo\n",p,lane,gidx); ++fails; }
                memcpy(&xb,&bhi,8); memcpy(&xs,&s[1],8);
                n1=((xb&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)&&(xb&0xfffffffffffffULL);
                n2=((xs&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)&&(xs&0xfffffffffffffULL);
                if (xb!=xs && !(n1&&n2)) { if(fails<6) printf("  MISMATCH prog %d lane %d F%d.hi\n",p,lane,gidx); ++fails; }
            };
            for (int k=0;k<randomx::RegisterCountFlt;++k) cmpF(k,    Fblo[k][lane],   Fbhi[k][lane],   nr.f[k]);
            for (int k=0;k<randomx::RegisterCountFlt;++k) cmpF(k+4,  Fblo[k+4][lane], Fbhi[k+4][lane], nr.e[k]);
        }
    }

    printf("== %s (%ld mismatches over %d programs x %d lanes) ==\n",
           fails==0?"ALL PASS":"FAILURES", fails, PROGRAMS, LANES);
    free(spB); free(spSnapFlat);
    return fails==0?0:1;
}

// ---- Stage 8 (skeleton): full batched hash vs a real full-mem reference VM ----
// Drives the reference chain-by-chain (run()+getRegisterFile()) and mirrors it
// batched, comparing the register file after EVERY program so a divergence
// localizes to the first bad chain; final 32-byte hash compared at the end.
// Heavy: builds the real FastMode (~2 GiB) dataset via xmrig Rx (run locally).
int verifyBatchedHash()
{
    _mm_setcsr(0x9FC0);
    RandomX_CurrentConfig.Apply();
    printf("== BatchedVm Stage 8: full batched hash (vs randomx_calculate_hash) ==\n");

    const uint32_t ITER    = RandomX_CurrentConfig.ProgramIterations;
    const uint32_t PC      = RandomX_CurrentConfig.ProgramCount;
    const uint64_t spBytes = RandomX_CurrentConfig.ScratchpadL3_Size;
    const uint64_t spWords = spBytes / 8;
    const int NHASH = 2;
    long fails = 0;

    Stage8Ctx ctx = stage8_setup();
    randomx_vm* vm = ctx.vm;
    const uint64_t* ds = ctx.ds;
    if(!vm){ printf("  8: vm setup failed\n"); return 1; }

    uint64_t* spB = (uint64_t*)malloc((size_t)LANES*spWords*8);
    if(!spB){ printf("  8: scratchpad alloc failed\n"); stage8_teardown(vm); return 1; }

    auto getSmallPos=[](uint64_t e)->uint64_t{ uint64_t exp=e>>59, man=e&((1ULL<<52)-1); exp+=1023; exp&=2047; exp<<=52; return exp|man; };
    auto bits2d=[](uint64_t b)->double{ double d; memcpy(&d,&b,8); return d; };
    auto d2bits=[](double d)->uint64_t{ uint64_t b; memcpy(&b,&d,8); return b; };

    for(int n=0; n<NHASH; ++n){
        alignas(16) uint64_t input[8]; for(int i=0;i<8;++i) input[i]=0x9e3779b97f4a7c15ULL*(uint64_t)(n+1)+(uint64_t)i;

        alignas(16) uint64_t tempHash[8];
        rx_blake2b_default(tempHash, sizeof(tempHash), input, sizeof(input));

        // batched scratchpad from the ORIGINAL seed, fresh copy per lane
        // (fillAes1Rx4 writes the evolved state back to its seed arg).
        for(int lane=0;lane<LANES;++lane){
            alignas(16) uint64_t sd[8]; memcpy(sd, tempHash, sizeof(tempHash));
            fillAes1Rx4<false>(sd, spBytes, spB + (size_t)lane*spWords);
        }

        fprintf(stderr, "[8] hash %d: initScratchpad\n", n);
        vm->initScratchpad(tempHash);   // reference scratchpad from same seed; mutates tempHash
        vm->resetRoundingMode();
        randomx::RegisterFile* r0 = vm->getRegisterFile();
        __m512i rV[IREGS];
        for(int k=0;k<IREGS;++k) rV[k]=_mm512_set1_epi64((long long)r0->r[k]);
        BFReg Fb[8]; for(int k=0;k<8;++k){ Fb[k].lo=_mm512_setzero_pd(); Fb[k].hi=_mm512_setzero_pd(); }
        int rmode[LANES]; for(int l=0;l<LANES;++l) rmode[l]=0;

        for(uint32_t chain=0; chain<PC; ++chain){
            // RandomX starts each program with a fresh register file (nreg.r zero-init,
            // only a-regs come from the program); integer registers do NOT carry.
            for(int k=0;k<IREGS;++k) rV[k]=_mm512_setzero_si512();
            for(int k=0;k<8;++k){ Fb[k].lo=_mm512_setzero_pd(); Fb[k].hi=_mm512_setzero_pd(); }
            randomx::Program program;
            alignas(16) uint64_t seedProg[8]; memcpy(seedProg, tempHash, sizeof(tempHash));
            fillAes4Rx4<false>(seedProg, 128 + RandomX_CurrentConfig.ProgramSize*8, &program);
            randomx::NativeRegisterFile nregT;
            static randomx::InstructionByteCode btT[512];
            randomx::BytecodeMachine bmT; bmT.compileProgram(program, btT, nregT);

            randomx::ProgramConfiguration cfg;
            auto staticExp=[&](uint64_t e){ uint64_t x=0x300; x|=(e>>(64-4))<<4; x<<=52; return x; };
            auto floatMask=[&](uint64_t e){ return (e & ((1ULL<<22)-1)) | staticExp(e); };
            cfg.eMask[0]=floatMask(program.getEntropy(14));
            cfg.eMask[1]=floatMask(program.getEntropy(15));
            uint64_t ar=program.getEntropy(12);
            cfg.readReg0=0+(ar&1); ar>>=1; cfg.readReg1=2+(ar&1); ar>>=1;
            cfg.readReg2=4+(ar&1); ar>>=1; cfg.readReg3=6+(ar&1);
            const uint32_t ma0=(uint32_t)(program.getEntropy(8) & CacheLineAlignMask);
            const uint32_t mx0=(uint32_t)(program.getEntropy(10));
            const uint64_t datasetOffset=(program.getEntropy(13) % (DatasetExtraItems+1)) * randomx::CacheLineSize;

            BFReg Ab[4];
            for(int k=0;k<4;++k){
                Ab[k].lo=_mm512_set1_pd(bits2d(getSmallPos(program.getEntropy(2*k))));
                Ab[k].hi=_mm512_set1_pd(bits2d(getSmallPos(program.getEntropy(2*k+1))));
            }
            FullProg fp; fp.count=(int)RandomX_CurrentConfig.ProgramSize;
            for(int i=0;i<fp.count;++i) translateFull(btT[i], nregT, fp.ins[i], true);

            __m512i maV=_mm512_set1_epi64((long long)ma0), mxV=_mm512_set1_epi64((long long)mx0);
            fprintf(stderr, "[8] chain %u: runBatchedExecute (dsoff=%llu)\n", chain, (unsigned long long)datasetOffset);
            runBatchedExecute(rV, Fb, Ab, spB, spWords, cfg.eMask, fp, maV, mxV,
                              (int)cfg.readReg0,(int)cfg.readReg1,(int)cfg.readReg2,(int)cfg.readReg3,
                              datasetOffset, ds, ~0ull, ITER, rmode);
            fprintf(stderr, "[8] chain %u: vm->run\n", chain);

            vm->run(tempHash);
            randomx::RegisterFile* rf = vm->getRegisterFile();

            uint64_t rOut[IREGS][LANES]; for(int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)rOut[k], rV[k]);
            double Fblo[8][LANES], Fbhi[8][LANES];
            for(int k=0;k<8;++k){ _mm512_storeu_pd(Fblo[k],Fb[k].lo); _mm512_storeu_pd(Fbhi[k],Fb[k].hi); }
            for(int lane=0;lane<LANES;++lane){
                for(int k=0;k<IREGS;++k)
                    if(rOut[k][lane]!=rf->r[k]){ if(fails<8) printf("  MISMATCH hash %d chain %u lane %d R%d\n",n,chain,lane,k); ++fails; }
                for(int k=0;k<4;++k){
                    if(d2bits(Fblo[k][lane])!=d2bits(rf->f[k].lo)){ if(fails<8) printf("  MISMATCH hash %d chain %u lane %d F%d.lo\n",n,chain,lane,k); ++fails; }
                    if(d2bits(Fbhi[k][lane])!=d2bits(rf->f[k].hi)){ if(fails<8) printf("  MISMATCH hash %d chain %u lane %d F%d.hi\n",n,chain,lane,k); ++fails; }
                    if(d2bits(Fblo[k+4][lane])!=d2bits(rf->e[k].lo)){ if(fails<8) printf("  MISMATCH hash %d chain %u lane %d F%d.lo\n",n,chain,lane,k+4); ++fails; }
                    if(d2bits(Fbhi[k+4][lane])!=d2bits(rf->e[k].hi)){ if(fails<8) printf("  MISMATCH hash %d chain %u lane %d F%d.hi\n",n,chain,lane,k+4); ++fails; }
                }
            }
            if(n==0){
                const uint64_t* refsp = (const uint64_t*)ctx.scratch;
                long spdiff=-1; for(uint64_t w=0; w<spWords; ++w) if(spB[w]!=refsp[w]){ spdiff=(long)w; break; }
                fprintf(stderr,"[8] chain0 scratchpad firstDiffWord=%ld (spWords=%llu)\n", spdiff, (unsigned long long)spWords);
                if(spdiff>=0) fprintf(stderr,"[8]   sp[%ld] batched=%016llx ref=%016llx\n", spdiff,
                                      (unsigned long long)spB[spdiff], (unsigned long long)refsp[spdiff]);
                for(int k=0;k<IREGS;++k) fprintf(stderr,"[8]   R%d batched=%016llx ref=%016llx\n", k,
                                      (unsigned long long)rOut[k][0], (unsigned long long)rf->r[k]);
            }
            if(chain+1<PC) rx_blake2b_default(tempHash, sizeof(tempHash), rf, sizeof(randomx::RegisterFile));
        }

        alignas(16) uint8_t refOut[32];
        vm->getFinalResult(refOut);

        uint64_t rOut[IREGS][LANES]; for(int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)rOut[k], rV[k]);
        double Fblo[8][LANES], Fbhi[8][LANES];
        for(int k=0;k<8;++k){ _mm512_storeu_pd(Fblo[k],Fb[k].lo); _mm512_storeu_pd(Fbhi[k],Fb[k].hi); }
        for(int lane=0;lane<LANES;++lane){
            randomx::RegisterFile rfl; memset(&rfl,0,sizeof(rfl));
            for(int k=0;k<IREGS;++k) rfl.r[k]=rOut[k][lane];
            for(int k=0;k<4;++k){ rfl.f[k].lo=Fblo[k][lane]; rfl.f[k].hi=Fbhi[k][lane];
                                  rfl.e[k].lo=Fblo[k+4][lane]; rfl.e[k].hi=Fbhi[k+4][lane]; }
            hashAes1Rx4<false>(spB + (size_t)lane*spWords, spBytes, &rfl.a);
            alignas(16) uint8_t out[32];
            rx_blake2b_default(out, 32, &rfl, sizeof(randomx::RegisterFile));
            if(memcmp(out, refOut, 32)!=0){ if(fails<8) printf("  MISMATCH hash %d lane %d final\n",n,lane); ++fails; }
        }
    }

    free(spB);
    stage8_teardown(vm);
    printf("== %s (%ld mismatches over %d hashes x %d lanes) ==\n",
           fails==0?"ALL PASS":"FAILURES", fails, NHASH, LANES);
    return fails==0?0:1;
}

int verifyProgramFull()        { return verifyProgramFullImpl(false, "6b"); }
int verifyProgramFullRounded() { return verifyProgramFullImpl(true,  "6c"); }

// ---- Stage 7: batched execute() loop vs a faithful per-lane transcription ----
// Oracle = reference executeBytecode wrapped in the exact vm_interpreted execute()
// framing; both sides share one reduced pseudo-random dataset (masked identically).
int verifyBatchedExecute()
{
    _mm_setcsr(0x9FC0);
    RandomX_CurrentConfig.Apply();
    printf("== BatchedVm Stage 7: batched execute() loop (vs execute() transcription) ==\n");

    const int      PROGRAMS = 8;
    const uint32_t ITER     = RandomX_CurrentConfig.ProgramIterations;
    const uint64_t spWords  = 2097152ull / 8;          // 2 MiB / lane
    const uint64_t dsWords  = 8ull * 1024 * 1024;      // 64 MiB reduced dataset
    const uint64_t dmask    = dsWords - 1;
    long fails = 0;
    rng_s = 0x2545F4914F6CDD1DULL ^ 0x0BADC0DE;

    uint64_t* spB   = (uint64_t*)malloc((size_t)LANES*spWords*8);
    uint64_t* spSn  = (uint64_t*)malloc((size_t)LANES*spWords*8);
    uint64_t* spL   = (uint64_t*)malloc((size_t)spWords*8);      // scalar working copy
    uint64_t* ds    = (uint64_t*)malloc((size_t)dsWords*8);
    if(!spB||!spSn||!spL||!ds){ printf("  7: alloc failed\n"); free(spB);free(spSn);free(spL);free(ds); return 1; }
    for(uint64_t i=0;i<dsWords;++i) ds[i]=rng();
    auto SNAP=[&](int l)->uint64_t*{ return spSn + (size_t)l*spWords; };

    const __m128d mant = _mm_castsi128_pd(_mm_set1_epi64x((long long)E_MANTISSA_MASK));
    const bool prefetchTweak = RandomX_CurrentConfig.Tweak_V2_PREFETCH;

    for(int p=0;p<PROGRAMS;++p){
        alignas(16) uint8_t seed[64]; for(int i=0;i<64;++i) seed[i]=(uint8_t)rng();
        randomx::Program program;
        fillAes4Rx4<false>(seed, 128 + RandomX_CurrentConfig.ProgramSize*8, &program);

        randomx::NativeRegisterFile nregT;
        static randomx::InstructionByteCode btT[512];
        randomx::BytecodeMachine bmT; bmT.compileProgram(program, btT, nregT);

        randomx::ProgramConfiguration cfg;
        auto staticExp=[&](uint64_t e){ uint64_t x=0x300; x|=(e>>(64-4))<<4; x<<=52; return x; };
        auto floatMask=[&](uint64_t e){ return (e & ((1ULL<<22)-1)) | staticExp(e); };
        cfg.eMask[0]=floatMask(program.getEntropy(14));
        cfg.eMask[1]=floatMask(program.getEntropy(15));
        uint64_t ar=program.getEntropy(12);
        cfg.readReg0=0+(ar&1); ar>>=1; cfg.readReg1=2+(ar&1); ar>>=1;
        cfg.readReg2=4+(ar&1); ar>>=1; cfg.readReg3=6+(ar&1);
        const uint32_t ma0=(uint32_t)(program.getEntropy(8) & CacheLineAlignMask);
        const uint32_t mx0=(uint32_t)(program.getEntropy(10));
        const uint64_t datasetOffset=(program.getEntropy(13) % (DatasetExtraItems+1)) * randomx::CacheLineSize;
        const __m128d emask = _mm_castsi128_pd(_mm_set_epi64x((long long)cfg.eMask[1],(long long)cfg.eMask[0]));

        FullProg fp; fp.count=(int)RandomX_CurrentConfig.ProgramSize;
        for(int i=0;i<fp.count;++i) translateFull(btT[i], nregT, fp.ins[i], /*cfround=*/true);

        uint64_t rSnap[IREGS][LANES];
        double aLo[4][LANES], aHi[4][LANES];
        for(int lane=0;lane<LANES;++lane){
            for(int k=0;k<IREGS;++k) rSnap[k][lane]=rng();
            for(int k=0;k<4;++k){ aLo[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0;
                                  aHi[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0; }
            for(uint64_t w=0;w<spWords;++w) SNAP(lane)[w]=rng();
        }

        uint64_t rB[IREGS][LANES];
        for(int k=0;k<IREGS;++k) for(int l=0;l<LANES;++l) rB[k][l]=rSnap[k][l];
        for(int lane=0;lane<LANES;++lane)
            for(uint64_t w=0;w<spWords;++w) spB[lane*spWords+w]=SNAP(lane)[w];
        BFReg Fb[8], Ab[4];
        for(int k=0;k<8;++k){ Fb[k].lo=_mm512_setzero_pd(); Fb[k].hi=_mm512_setzero_pd(); }
        for(int k=0;k<4;++k){ Ab[k].lo=_mm512_loadu_pd(aLo[k]); Ab[k].hi=_mm512_loadu_pd(aHi[k]); }
        __m512i maV=_mm512_set1_epi64((long long)ma0), mxV=_mm512_set1_epi64((long long)mx0);
        __m512i rV[IREGS];
        for(int k=0;k<IREGS;++k) rV[k]=_mm512_loadu_si512((const void*)rB[k]);
        int rmode[LANES]; for(int l=0;l<LANES;++l) rmode[l]=0;
        runBatchedExecute(rV, Fb, Ab, spB, spWords, cfg.eMask, fp, maV, mxV,
                          (int)cfg.readReg0,(int)cfg.readReg1,(int)cfg.readReg2,(int)cfg.readReg3,
                          datasetOffset, ds, dmask, ITER, rmode);
        for(int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)rB[k], rV[k]);
        double Fblo[8][LANES], Fbhi[8][LANES];
        for(int k=0;k<8;++k){ _mm512_storeu_pd(Fblo[k],Fb[k].lo); _mm512_storeu_pd(Fbhi[k],Fb[k].hi); }

        for(int lane=0;lane<LANES;++lane){
            randomx::NativeRegisterFile nr;
            static randomx::InstructionByteCode bt[512];
            randomx::BytecodeMachine bm; bm.compileProgram(program, bt, nr);
            for(int k=0;k<IREGS;++k) nr.r[k]=rSnap[k][lane];
            for(int k=0;k<4;++k) nr.a[k]=_mm_set_pd(aHi[k][lane],aLo[k][lane]);
            for(uint64_t w=0;w<spWords;++w) spL[w]=SNAP(lane)[w];
            uint8_t* sp=(uint8_t*)spL;
            uint32_t ma=ma0, mx=mx0, spAddr0=mx0, spAddr1=ma0;
            rx_reset_float_state();
            for(uint32_t ic=0; ic<ITER; ++ic){
                uint64_t spMix=nr.r[cfg.readReg0]^nr.r[cfg.readReg1];
                spAddr0=(spAddr0^(uint32_t)spMix)&(uint32_t)ScratchpadL3Mask64;
                spAddr1=(spAddr1^(uint32_t)(spMix>>32))&(uint32_t)ScratchpadL3Mask64;
                for(int i=0;i<IREGS;++i) nr.r[i]^=load64(sp+spAddr0+8*i);
                for(int i=0;i<4;++i) nr.f[i]=rx_cvt_packed_int_vec_f128(sp+spAddr1+8*i);
                for(int i=0;i<4;++i){ rx_vec_f128 x=rx_cvt_packed_int_vec_f128(sp+spAddr1+8*(4+i));
                                      nr.e[i]=_mm_or_pd(_mm_and_pd(x,mant),emask); }
                randomx::BytecodeMachine::executeBytecode(bt, sp, cfg);
                uint64_t readPtr=datasetOffset+(ma & CacheLineAlignMask);
                uint32_t mp=(uint32_t)(nr.r[cfg.readReg2]^nr.r[cfg.readReg3]);
                if(prefetchTweak) ma^=mp; else mx^=mp;
                for(int i=0;i<IREGS;++i) nr.r[i]^=ds[((readPtr>>3)+i)&dmask];
                { uint32_t t=mx; mx=ma; ma=t; }
                for(int i=0;i<IREGS;++i) store64(sp+spAddr1+8*i, nr.r[i]);
                for(int i=0;i<4;++i) nr.f[i]=rx_xor_vec_f128(nr.f[i],nr.e[i]);
                for(int i=0;i<4;++i) rx_store_vec_f128((double*)(sp+spAddr0+16*i), nr.f[i]);
                spAddr0=0; spAddr1=0;
            }

            for(int k=0;k<IREGS;++k)
                if(rB[k][lane]!=nr.r[k]){ if(fails<8) printf("  MISMATCH prog %d lane %d R%d\n",p,lane,k); ++fails; }
            auto cmpF=[&](int gidx,double blo,double bhi,rx_vec_f128 sv){
                alignas(16) double s[2]; _mm_store_pd(s,sv);
                uint64_t xb,xs; memcpy(&xb,&blo,8); memcpy(&xs,&s[0],8);
                if(xb!=xs){ if(fails<8) printf("  MISMATCH prog %d lane %d F%d.lo\n",p,lane,gidx); ++fails; }
                memcpy(&xb,&bhi,8); memcpy(&xs,&s[1],8);
                if(xb!=xs){ if(fails<8) printf("  MISMATCH prog %d lane %d F%d.hi\n",p,lane,gidx); ++fails; }
            };
            for(int k=0;k<4;++k) cmpF(k,   Fblo[k][lane],   Fbhi[k][lane],   nr.f[k]);
            for(int k=0;k<4;++k) cmpF(k+4, Fblo[k+4][lane], Fbhi[k+4][lane], nr.e[k]);
        }
    }

    printf("== %s (%ld mismatches over %d programs x %d lanes x %u iters) ==\n",
           fails==0?"ALL PASS":"FAILURES", fails, PROGRAMS, LANES, ITER);
    free(spB); free(spSn); free(spL); free(ds);
    return fails==0?0:1;
}

#if defined(__GNUC__)
#   pragma GCC pop_options
#endif

} // namespace batchedx
} // namespace xmrig
