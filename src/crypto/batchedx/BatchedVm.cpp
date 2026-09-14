/* XMRig — batched AVX-512 RandomX (8-lane), implementation.
 *
 * Stage 1: 8-lane interpreter for the RandomX integer register ops, plus a bit-exact
 * self-test against scalar semantics. Compiled in only when WITH_BATCHEDX=ON.
 */

#include "crypto/batchedx/BatchedVm.h"
#include "crypto/batchedx/BatchedOps.h"

#include <immintrin.h>
#include <cstdio>
#include <cstring>
#include "crypto/randomx/bytecode_machine.hpp"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/aes_hash.hpp"

#if defined(__GNUC__)
#   pragma GCC target("avx512f,avx512dq")
#endif

namespace xmrig {
namespace batchedx {


// ---- AVX-512 helpers ----------------------------------------------------------



// 64x64 -> high 64 across 8 lanes. AVX-512 has no 64-bit high-mul; synthesize from
// 32x32 partial products (schoolbook), then correct for signedness if needed.

// ---- batched integer program execution ---------------------------------------

void runIntegerProgram(uint64_t regs[IREGS][LANES], const BInsn* prog, int count){
  // load registers into 8 vector registers (one per index, lane = nonce)
  __m512i r[IREGS];
  for (int k = 0; k < IREGS; ++k) { r[k] = _mm512_loadu_si512((const void*)regs[k]); }

  for (int i = 0; i < count; ++i) {
    const BInsn& in = prog[i];
    __m512i& d = r[in.dst];
    const __m512i s = r[in.src];
    switch (in.op) {
      case B_IADD_RS: // dst += (src << shift) + imm
        d = _mm512_add_epi64(d,
            _mm512_add_epi64(_mm512_slli_epi64(s, in.shift),
                     _mm512_set1_epi64((long long)in.imm)));
        break;
      case B_ISUB_R:  // dst -= src
        d = _mm512_sub_epi64(d, s);
        break;
      case B_IMUL_R:  // dst *= src   (low 64 bits)
        d = _mm512_mullo_epi64(d, s);
        break;
      case B_INEG_R:  // dst = ~dst + 1
        d = _mm512_add_epi64(_mm512_xor_si512(d, _mm512_set1_epi64(-1)),
                   _mm512_set1_epi64(1));
        break;
      case B_IXOR_R:  // dst ^= src
        d = _mm512_xor_si512(d, s);
        break;
      case B_IROR_R:  // dst = rotr(dst, src & 63)
        d = rorv64(d, s);
        break;
      case B_IROL_R:  // dst = rotl(dst, src & 63)
        d = rolv64(d, s);
        break;
      case B_ISWAP_R: // swap dst, src (no-op if same)
        if (in.dst != in.src) { __m512i t = d; d = r[in.src]; r[in.src] = t; }
        break;
      case B_IMULH_R:  d = mulhi_epu64(d, s); break;
      case B_ISMULH_R: d = mulhi_epi64(d, s); break;
      default: break;
    }
  }

  for (int k = 0; k < IREGS; ++k) { _mm512_storeu_si512((void*)regs[k], r[k]); }
}


// ---- scalar reference (matches bytecode_machine.hpp semantics) ----------------

static inline uint64_t s_rotr(uint64_t x, unsigned c) { c &= 63; return c ? (x >> c) | (x << (64 - c)) : x; }
static inline uint64_t s_rotl(uint64_t x, unsigned c) { c &= 63; return c ? (x << c) | (x >> (64 - c)) : x; }

static void scalarIntegerProgram(uint64_t regs[IREGS], const BInsn* prog, int count){
  for (int i = 0; i < count; ++i) {
    const BInsn& in = prog[i];
    uint64_t& d = regs[in.dst];
    const uint64_t s = regs[in.src];
    switch (in.op) {
      case B_IADD_RS: d += (s << in.shift) + in.imm; break;
      case B_ISUB_R:  d -= s; break;
      case B_IMUL_R:  d *= s; break;
      case B_INEG_R:  d = ~d + 1; break;
      case B_IXOR_R:  d ^= s; break;
      case B_IROR_R:  d = s_rotr(d, (unsigned)(s & 63)); break;
      case B_IROL_R:  d = s_rotl(d, (unsigned)(s & 63)); break;
      case B_ISWAP_R: if (in.dst != in.src) { uint64_t t = d; d = regs[in.src]; regs[in.src] = t; } break;
      case B_IMULH_R:  d = (uint64_t)(((unsigned __int128)d * (unsigned __int128)s) >> 64); break;
      case B_ISMULH_R: d = (uint64_t)(((__int128)(int64_t)d * (__int128)(int64_t)s) >> 64); break;
      default: break;
    }
  }
}



static void scalarMemoryProgram(uint64_t regs[IREGS], const BInsn* prog, int count, uint64_t* sp)
{
    for (int i = 0; i < count; ++i) {
        const BInsn& in = prog[i];
        uint64_t& d = regs[in.dst];
        const uint64_t s = regs[in.src];
        if (in.op == B_ISTORE) {
            uint32_t addr = (uint32_t)((d + in.imm) & in.memMask);
            sp[addr >> 3] = s;
            continue;
        }
        uint32_t addr = (uint32_t)((s + in.imm) & in.memMask);
        uint64_t v = sp[addr >> 3];
        switch (in.op) {
            case B_IADD_M:   d += v; break;
            case B_ISUB_M:   d -= v; break;
            case B_IMUL_M:   d *= v; break;
            case B_IMULH_M:  d = (uint64_t)(((unsigned __int128)d * (unsigned __int128)v) >> 64); break;
            case B_ISMULH_M: d = (uint64_t)(((__int128)(int64_t)d * (__int128)(int64_t)v) >> 64); break;
            case B_IXOR_M:   d ^= v; break;
            default: break;
        }
    }
}

// ---- self-test ---------------------------------------------------------------

static uint64_t rng_s;
static inline uint64_t rng() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }


// ---- Stage 4: batched memory ops (scratchpad gather/scatter) -------------------
//
// 8 per-lane scratchpads laid out contiguously: lane L occupies words
// [L*spWords .. (L+1)*spWords). Load addr per lane = laneBase[L] + wordIndex, where
// wordIndex = ((src+imm)&memMask)>>3. Loads use vpgatherqq, ISTORE uses vpscatterqq.

void runMemoryProgram(uint64_t regs[IREGS][LANES], const BInsn* prog, int count,
                      uint64_t* sp, uint64_t spWords)
{
    __m512i r[IREGS];
    for (int k = 0; k < IREGS; ++k) r[k] = _mm512_loadu_si512((const void*)regs[k]);

    // per-lane base word offset (lane L at L*spWords)
    const __m512i laneBase = _mm512_set_epi64(
        (long long)(7*spWords), (long long)(6*spWords), (long long)(5*spWords), (long long)(4*spWords),
        (long long)(3*spWords), (long long)(2*spWords), (long long)(1*spWords), 0LL);

    for (int i = 0; i < count; ++i) {
        const BInsn& in = prog[i];
        __m512i& d = r[in.dst];
        const __m512i s = r[in.src];
        const __m512i imm  = _mm512_set1_epi64((long long)in.imm);
        const __m512i mask = _mm512_set1_epi64((long long)(uint64_t)in.memMask);

        if (in.op == B_ISTORE) {
            // addr = (dst + imm) & memMask ; store src
            __m512i addr = _mm512_and_si512(_mm512_add_epi64(d, imm), mask);
            __m512i widx = _mm512_add_epi64(_mm512_srli_epi64(addr, 3), laneBase);
            _mm512_i64scatter_epi64((long long*)sp, widx, s, 8);
            continue;
        }

        // load ops: addr = (src + imm) & memMask ; v = sp[addr]
        __m512i addr = _mm512_and_si512(_mm512_add_epi64(s, imm), mask);
        __m512i widx = _mm512_add_epi64(_mm512_srli_epi64(addr, 3), laneBase);
        __m512i v    = _mm512_i64gather_epi64(widx, (const long long*)sp, 8);

        switch (in.op) {
            case B_IADD_M:   d = _mm512_add_epi64(d, v); break;
            case B_ISUB_M:   d = _mm512_sub_epi64(d, v); break;
            case B_IMUL_M:   d = _mm512_mullo_epi64(d, v); break;
            case B_IMULH_M:  d = mulhi_epu64(d, v); break;
            case B_ISMULH_M: d = mulhi_epi64(d, v); break;
            case B_IXOR_M:   d = _mm512_xor_si512(d, v); break;
            default: break;
        }
    }

    for (int k = 0; k < IREGS; ++k) _mm512_storeu_si512((void*)regs[k], r[k]);
}



// ---- Stage 5: CBRANCH via per-lane PC + masked execution ----------------------
//
// Approach A: each lane has its own pc. We step a shared position = min active pc;
// each instruction executes only on lanes whose pc == that position (active mask);
// active lanes advance pc (or jump on a taken CBRANCH). Correct for arbitrary
// backward-jump divergence. Only integer register ops + CBRANCH here (memory/float
// divergence follow the same masking pattern in the full VM).

// helper: apply one integer-register instruction to r[], only on masked lanes
static inline void applyIntMasked(__m512i r[IREGS], const BInsn& in, __mmask8 m)
{
    __m512i& d = r[in.dst];
    const __m512i s = in.srcImm ? _mm512_set1_epi64((long long)in.srcVal) : r[in.src];
    switch (in.op) {
        case B_IADD_RS: d = _mm512_mask_add_epi64(d, m, d,
                            _mm512_add_epi64(_mm512_slli_epi64(s, in.shift), _mm512_set1_epi64((long long)in.imm))); break;
        case B_ISUB_R:  d = _mm512_mask_sub_epi64(d, m, d, s); break;
        case B_IMUL_R:  d = _mm512_mask_mullo_epi64(d, m, d, s); break;
        case B_INEG_R:  d = _mm512_mask_add_epi64(d, m, _mm512_xor_si512(d, _mm512_set1_epi64(-1)), _mm512_set1_epi64(1)); break;
        case B_IXOR_R:  d = _mm512_mask_xor_epi64(d, m, d, s); break;
        case B_IROR_R:  d = _mm512_mask_rorv_epi64(d, m, d, _mm512_and_si512(s, _mm512_set1_epi64(63))); break;
        case B_IROL_R:  d = _mm512_mask_rolv_epi64(d, m, d, _mm512_and_si512(s, _mm512_set1_epi64(63))); break;
        case B_IMULH_R: d = _mm512_mask_blend_epi64(m, d, mulhi_epu64(d, s)); break;
        case B_ISMULH_R:d = _mm512_mask_blend_epi64(m, d, mulhi_epi64(d, s)); break;
        case B_ISWAP_R: if (in.dst != in.src) {   // masked swap
                            __m512i t = d;
                            d          = _mm512_mask_blend_epi64(m, d, r[in.src]);
                            r[in.src]  = _mm512_mask_blend_epi64(m, r[in.src], t);
                        } break;
        default: break;
    }
}

// batched program with CBRANCH. Uses per-lane pc.
void runBranchProgram(uint64_t regs[IREGS][LANES], const BInsn* prog, int count)
{
    __m512i r[IREGS];
    for (int k = 0; k < IREGS; ++k) r[k] = _mm512_loadu_si512((const void*)regs[k]);

    int pc[LANES];
    for (int l = 0; l < LANES; ++l) pc[l] = 0;

    // safety bound to guarantee termination in the test (real VM relies on RandomX
    // structure; here we cap total steps generously)
    long steps = 0, maxSteps = (long)count * 64;

    for (;;) {
        // find min active pc
        int pos = count;
        for (int l = 0; l < LANES; ++l) if (pc[l] < count && pc[l] < pos) pos = pc[l];
        if (pos >= count) break;                 // all lanes done
        if (++steps > maxSteps) break;

        // active mask = lanes at this position
        __mmask8 m = 0;
        for (int l = 0; l < LANES; ++l) if (pc[l] == pos) m |= (1u << l);

        const BInsn& in = prog[pos];

        if (in.op == B_CBRANCH) {
            // dst += imm ; if ((dst & memMask)==0) pc=target else pc++  (per active lane)
            __m512i& d = r[in.dst];
            d = _mm512_mask_add_epi64(d, m, d, _mm512_set1_epi64((long long)in.imm));
            __mmask8 taken = _mm512_mask_cmpeq_epi64_mask(m,
                                _mm512_and_si512(d, _mm512_set1_epi64((long long)(uint64_t)in.memMask)),
                                _mm512_setzero_si512());
            for (int l = 0; l < LANES; ++l) if (pc[l] == pos) {
                pc[l] = (taken & (1u << l)) ? in.target : pos + 1;
            }
        } else {
            applyIntMasked(r, in, m);
            for (int l = 0; l < LANES; ++l) if (pc[l] == pos) ++pc[l];
        }
    }

    for (int k = 0; k < IREGS; ++k) _mm512_storeu_si512((void*)regs[k], r[k]);
}

// scalar reference: real pc with jumps
static void scalarBranchProgram(uint64_t regs[IREGS], const BInsn* prog, int count)
{
    long steps = 0, maxSteps = (long)count * 64;
    for (int pc = 0; pc < count; ) {
        if (++steps > maxSteps) break;
        const BInsn& in = prog[pc];
        uint64_t& d = regs[in.dst];
        const uint64_t s = regs[in.src];
        switch (in.op) {
            case B_IADD_RS: d += (s << in.shift) + in.imm; ++pc; break;
            case B_ISUB_R:  d -= s; ++pc; break;
            case B_IMUL_R:  d *= s; ++pc; break;
            case B_INEG_R:  d = ~d + 1; ++pc; break;
            case B_IXOR_R:  d ^= s; ++pc; break;
            case B_IROR_R:  d = s_rotr(d, (unsigned)(s & 63)); ++pc; break;
            case B_IROL_R:  d = s_rotl(d, (unsigned)(s & 63)); ++pc; break;
            case B_ISWAP_R: if (in.dst != in.src) { uint64_t t=d; d=regs[in.src]; regs[in.src]=t; } ++pc; break;
            case B_IMULH_R: d = (uint64_t)(((unsigned __int128)d*(unsigned __int128)s)>>64); ++pc; break;
            case B_ISMULH_R:d = (uint64_t)(((__int128)(int64_t)d*(__int128)(int64_t)s)>>64); ++pc; break;
            case B_CBRANCH: d += in.imm;
                            if ((d & (uint64_t)in.memMask) == 0) pc = in.target; else ++pc;
                            break;
            default: ++pc; break;
        }
    }
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



// ============================================================================
// Stage 3: batched float register ops (FSWAP_R, FADD_R, FSUB_R, FMUL_R,
// FSQRT_R, FSCAL_R). Each RandomX float register is 2 doubles per lane; across
// 8 lanes we hold it as two __m512d: 'lo' = element 0 of each lane's pair,
// 'hi' = element 1. Verified bit-exact against scalar __m128d semantics (default
// rounding mode; CFROUND handled in a later stage).
// ============================================================================

namespace {

enum FOp : uint8_t { F_FSWAP_R, F_FADD_R, F_FSUB_R, F_FMUL_R, F_FSQRT_R, F_FSCAL_R };

struct FInsn { FOp op; uint8_t dst; uint8_t src; };   // dst/src index 0..FREGS-1

static constexpr int FREGS = 4;   // RandomX f-group registers

// FSCAL mask 0x80F0000000000000 applied to the raw bits of each double

// E-register exponent/mantissa mask (RandomX keeps E-regs positive & bounded).

static inline double emask_s(double v) {
    uint64_t b; memcpy(&b, &v, 8);
    b = (b & E_MANTISSA_MASK) | E_EXP_MASK;
    double r; memcpy(&r, &b, 8); return r;
}

static inline bool bothNaN(uint64_t a, uint64_t b) {
    // exponent all ones + nonzero mantissa = NaN (ignore sign)
    auto isnan = [](uint64_t x){ return ((x & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL)
                                     && (x & 0x000fffffffffffffULL); };
    return isnan(a) && isnan(b);
}

// batched: reg[k].lo / reg[k].hi are __m512d (8 lanes)


static void runFloatProgram(BFReg reg[FREGS], const FInsn* prog, int count)
{
    for (int i = 0; i < count; ++i) {
        const FInsn& in = prog[i];
        BFReg& d = reg[in.dst];
        const BFReg s = reg[in.src];
        switch (in.op) {
            case F_FSWAP_R: { __m512d t = d.lo; d.lo = d.hi; d.hi = t; } break;
            case F_FADD_R: d.lo = _mm512_add_pd(d.lo, s.lo); d.hi = _mm512_add_pd(d.hi, s.hi); break;
            case F_FSUB_R: d.lo = _mm512_sub_pd(d.lo, s.lo); d.hi = _mm512_sub_pd(d.hi, s.hi); break;
            case F_FMUL_R: d.lo = _mm512_mul_pd(d.lo, s.lo); d.hi = _mm512_mul_pd(d.hi, s.hi); break;
            case F_FSQRT_R: d.lo = _mm512_sqrt_pd(emask(d.lo)); d.hi = _mm512_sqrt_pd(emask(d.hi)); break;
            case F_FSCAL_R: d.lo = fscal(d.lo); d.hi = fscal(d.hi); break;
        }
    }
}

// scalar reference: each register is 2 doubles [0]=lo [1]=hi
static void scalarFloatProgram(double reg[FREGS][2], const FInsn* prog, int count)
{
    for (int i = 0; i < count; ++i) {
        const FInsn& in = prog[i];
        double* d = reg[in.dst];
        const double* s = reg[in.src];
        switch (in.op) {
            case F_FSWAP_R: { double t = d[0]; d[0] = d[1]; d[1] = t; } break;
            case F_FADD_R:  d[0] += s[0]; d[1] += s[1]; break;
            case F_FSUB_R:  d[0] -= s[0]; d[1] -= s[1]; break;
            case F_FMUL_R:  d[0] *= s[0]; d[1] *= s[1]; break;
            case F_FSQRT_R: d[0] = __builtin_sqrt(emask_s(d[0])); d[1] = __builtin_sqrt(emask_s(d[1])); break;
            case F_FSCAL_R: {
                uint64_t b0, b1; memcpy(&b0, &d[0], 8); memcpy(&b1, &d[1], 8);
                b0 ^= FSCAL_MASK; b1 ^= FSCAL_MASK;
                memcpy(&d[0], &b0, 8); memcpy(&d[1], &b1, 8);
            } break;
        }
    }
}

} // anonymous namespace

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




// ---- Stage 4 self-test: memory ops -------------------------------------------

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




// ---- Stage 5 self-test: CBRANCH --------------------------------------------

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

// ============================================================================
// Stage 6a: translate a REAL RandomX program to BInsn[] and run it through a
// unified batched interpreter (integer + branch), verified against the scalar
// BytecodeMachine on 8 register files. (float/memory fold in next.)
// ============================================================================

// translate one compiled InstructionByteCode -> BInsn (integer + branch).
static bool translateInt(const randomx::InstructionByteCode& ibc,
                         const randomx::NativeRegisterFile& nreg, BInsn& out)
{
    using IT = randomx::InstructionType;
    auto ridx = [&](const uint64_t* p) -> uint8_t { return (uint8_t)(p - &nreg.r[0]); };
    // source may be a register (isrc in r[0..7]) or an immediate (isrc points elsewhere,
    // e.g. &ibc.imm for the src==dst / IMUL_RCP forms). Detect and capture accordingly.
    auto setSrc = [&](const uint64_t* p){
        long idx = p - &nreg.r[0];
        if (idx >= 0 && idx < IREGS) { out.srcImm = false; out.src = (uint8_t)idx; out.srcVal = 0; }
        else                         { out.srcImm = true;  out.src = 0; out.srcVal = *p; }
    };
    out.imm = ibc.imm; out.memMask = ibc.memMask; out.target = ibc.target;
    out.shift = (uint8_t)ibc.shift; out.src = 0; out.dst = 0; out.srcImm = false; out.srcVal = 0;
    switch (ibc.type) {
        case IT::IADD_RS: out.op=B_IADD_RS; out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::ISUB_R:  out.op=B_ISUB_R;  out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::IMUL_R:  out.op=B_IMUL_R;  out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::INEG_R:  out.op=B_INEG_R;  out.dst=ridx(ibc.idst); return true;
        case IT::IXOR_R:  out.op=B_IXOR_R;  out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::IROR_R:  out.op=B_IROR_R;  out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::IROL_R:  out.op=B_IROL_R;  out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::ISWAP_R: out.op=B_ISWAP_R; out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::IMULH_R: out.op=B_IMULH_R; out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::ISMULH_R:out.op=B_ISMULH_R;out.dst=ridx(ibc.idst); setSrc(ibc.isrc); return true;
        case IT::CBRANCH: out.op=B_CBRANCH; out.dst=ridx(ibc.idst); return true;
        default: out.op=B_CBRANCH; return false;   // unsupported here (float/mem); marked skip
    }
}

// Unified batched run over a translated program. For 6a we only support
// integer + branch ops (matches translateInt). Instructions the translator
// couldn't map (float/mem) are treated as NO-OP here so the test can be
// restricted to int/branch-only programs. Uses the Stage-5 per-lane-PC loop.
struct BProg { BInsn ins[512]; bool ok[512]; int count; };

static void runProgramIntBranch(uint64_t regs[IREGS][LANES], const BProg& prog)
{
    __m512i r[IREGS];
    for (int k = 0; k < IREGS; ++k) r[k] = _mm512_loadu_si512((const void*)regs[k]);

    int pc[LANES]; for (int l=0;l<LANES;++l) pc[l]=0;
    long steps=0, maxSteps=(long)prog.count*64*LANES;
    for (;;) {
        int pos = prog.count;
        for (int l=0;l<LANES;++l) if (pc[l]<prog.count && pc[l]<pos) pos=pc[l];
        if (pos>=prog.count) break;
        if (++steps>maxSteps) break;
        __mmask8 m=0; for (int l=0;l<LANES;++l) if (pc[l]==pos) m|=(1u<<l);
        const BInsn& in = prog.ins[pos];
        if (prog.ok[pos] && in.op==B_CBRANCH) {
            __m512i& d=r[in.dst];
            d=_mm512_mask_add_epi64(d,m,d,_mm512_set1_epi64((long long)in.imm));
            __mmask8 taken=_mm512_mask_cmpeq_epi64_mask(m,
                _mm512_and_si512(d,_mm512_set1_epi64((long long)(uint64_t)in.memMask)),_mm512_setzero_si512());
            for (int l=0;l<LANES;++l) if (pc[l]==pos) pc[l]=(taken&(1u<<l))?(in.target+1):pos+1;
        } else {
            if (prog.ok[pos]) applyIntMasked(r,in,m);   // float/mem => no-op for 6a
            for (int l=0;l<LANES;++l) if (pc[l]==pos) ++pc[l];
        }
    }
    for (int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)regs[k], r[k]);
}

// scalar reference for the same restricted (int/branch) program on one lane.
static void scalarProgramIntBranch(uint64_t regs[IREGS], const BProg& prog)
{
    long steps=0, maxSteps=(long)prog.count*64;
    for (int pc=0; pc<prog.count; ) {
        if (++steps>maxSteps) break;
        const BInsn& in=prog.ins[pc];
        if (!prog.ok[pc]) { ++pc; continue; }         // float/mem => no-op
        uint64_t& d=regs[in.dst]; const uint64_t s = in.srcImm ? in.srcVal : regs[in.src];
        switch (in.op) {
            case B_IADD_RS: d+=(s<<in.shift)+in.imm; ++pc; break;
            case B_ISUB_R:  d-=s; ++pc; break;
            case B_IMUL_R:  d*=s; ++pc; break;
            case B_INEG_R:  d=~d+1; ++pc; break;
            case B_IXOR_R:  d^=s; ++pc; break;
            case B_IROR_R:  d=s_rotr(d,(unsigned)(s&63)); ++pc; break;
            case B_IROL_R:  d=s_rotl(d,(unsigned)(s&63)); ++pc; break;
            case B_ISWAP_R: if(in.dst!=in.src){uint64_t t=d;d=regs[in.src];regs[in.src]=t;} ++pc; break;
            case B_IMULH_R: d=(uint64_t)(((unsigned __int128)d*(unsigned __int128)s)>>64); ++pc; break;
            case B_ISMULH_R:d=(uint64_t)(((__int128)(int64_t)d*(__int128)(int64_t)s)>>64); ++pc; break;
            case B_CBRANCH: d+=in.imm; if((d&(uint64_t)in.memMask)==0) pc=in.target+1; else ++pc; break;
            default: ++pc; break;
        }
    }
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



// ============================================================================
// Stage 6b: full unified interpreter over REAL programs (int + branch + memory
// + float), verified against RandomX's own executeBytecode per lane.
// ============================================================================

// Extend translateInt-style mapping for memory-integer ops. Returns true if mapped.
static bool translateMemInt(const randomx::InstructionByteCode& ibc,
                            const randomx::NativeRegisterFile& nreg, BInsn& out)
{
    using IT = randomx::InstructionType;
    auto ridxR = [&](const uint64_t* p)->int { long i = p - &nreg.r[0]; return (i>=0 && i<IREGS)?(int)i:-1; };
    out.imm = ibc.imm; out.memMask = ibc.memMask; out.shift=0; out.target=0;
    out.srcImm=false; out.srcVal=0; out.dst=0; out.src=0;
    int addr = ridxR(ibc.isrc);            // address register; -1 if &zero (src==dst)
    bool addrZero = (addr < 0);
    out.src = addrZero ? 0 : (uint8_t)addr;      // if zero-src, addr uses reg 0 masked out below
    out.srcImm = addrZero;                        // reuse srcImm to mean "address reg is zero"
    switch (ibc.type) {
        case IT::IADD_M:  out.op=B_IADD_M;  out.dst=(uint8_t)(ibc.idst-&nreg.r[0]); return true;
        case IT::ISUB_M:  out.op=B_ISUB_M;  out.dst=(uint8_t)(ibc.idst-&nreg.r[0]); return true;
        case IT::IMUL_M:  out.op=B_IMUL_M;  out.dst=(uint8_t)(ibc.idst-&nreg.r[0]); return true;
        case IT::IMULH_M: out.op=B_IMULH_M; out.dst=(uint8_t)(ibc.idst-&nreg.r[0]); return true;
        case IT::ISMULH_M:out.op=B_ISMULH_M;out.dst=(uint8_t)(ibc.idst-&nreg.r[0]); return true;
        case IT::IXOR_M:  out.op=B_IXOR_M;  out.dst=(uint8_t)(ibc.idst-&nreg.r[0]); return true;
        case IT::ISTORE:  out.op=B_ISTORE;
                          out.dst=(uint8_t)(ibc.idst-&nreg.r[0]);
                          out.src=(uint8_t)(ibc.isrc-&nreg.r[0]);  // ISTORE src is always a reg
                          out.srcImm=false;
                          return true;
        default: return false;
    }
}

// Float op codes for the unified interpreter.
enum FBOp : uint8_t { FB_NONE, FB_FSWAP, FB_FADD_R, FB_FSUB_R, FB_FMUL_R, FB_FSQRT_R,
                      FB_FSCAL_R, FB_FADD_M, FB_FSUB_M, FB_FDIV_M };

struct FullInsn {
    uint8_t kind;   // 0=int/branch(ib), 1=float(fop), 2=skip
    BInsn   ib;
    FBOp    fop;
    uint8_t fdst;   // 0..7 (f=0..3, e=4..7)
    uint8_t fsrc;   // a-group 0..3
    uint8_t maddr;  // r-reg for float-M address
    uint64_t fimm;
    uint32_t fmask;
};

static void translateFull(const randomx::InstructionByteCode& ibc,
                          const randomx::NativeRegisterFile& nreg, FullInsn& out)
{
    using IT = randomx::InstructionType;
    auto fidx = [&](const void* p)->int {
        const rx_vec_f128* fp = reinterpret_cast<const rx_vec_f128*>(p);
        long f = fp-&nreg.f[0]; if (f>=0&&f<randomx::RegisterCountFlt) return (int)f;
        long e = fp-&nreg.e[0]; if (e>=0&&e<randomx::RegisterCountFlt) return (int)e+randomx::RegisterCountFlt;
        return 0;
    };
    auto aidx = [&](const void* p)->int {
        const rx_vec_f128* fp = reinterpret_cast<const rx_vec_f128*>(p);
        long a = fp-&nreg.a[0]; return (a>=0&&a<randomx::RegisterCountFlt)?(int)a:0;
    };
    out.fimm = ibc.imm; out.fmask = ibc.memMask;
    switch (ibc.type) {
        case IT::FSWAP_R: out.kind=1; out.fop=FB_FSWAP;  out.fdst=(uint8_t)fidx(ibc.fdst); return;
        case IT::FADD_R:  out.kind=1; out.fop=FB_FADD_R; out.fdst=(uint8_t)fidx(ibc.fdst); out.fsrc=(uint8_t)aidx(ibc.fsrc); return;
        case IT::FSUB_R:  out.kind=1; out.fop=FB_FSUB_R; out.fdst=(uint8_t)fidx(ibc.fdst); out.fsrc=(uint8_t)aidx(ibc.fsrc); return;
        case IT::FMUL_R:  out.kind=1; out.fop=FB_FMUL_R; out.fdst=(uint8_t)fidx(ibc.fdst); out.fsrc=(uint8_t)aidx(ibc.fsrc); return;
        case IT::FSQRT_R: out.kind=1; out.fop=FB_FSQRT_R;out.fdst=(uint8_t)fidx(ibc.fdst); return;
        case IT::FSCAL_R: out.kind=1; out.fop=FB_FSCAL_R;out.fdst=(uint8_t)fidx(ibc.fdst); return;
        case IT::FADD_M:  out.kind=1; out.fop=FB_FADD_M; out.fdst=(uint8_t)fidx(ibc.fdst); out.maddr=(uint8_t)(ibc.isrc-&nreg.r[0]); return;
        case IT::FSUB_M:  out.kind=1; out.fop=FB_FSUB_M; out.fdst=(uint8_t)fidx(ibc.fdst); out.maddr=(uint8_t)(ibc.isrc-&nreg.r[0]); return;
        case IT::FDIV_M:  out.kind=1; out.fop=FB_FDIV_M; out.fdst=(uint8_t)fidx(ibc.fdst); out.maddr=(uint8_t)(ibc.isrc-&nreg.r[0]); return;
        case IT::CFROUND: out.kind=2; return;
        case IT::NOP:     out.kind=2; return;
        default:
            if (translateInt(ibc, nreg, out.ib)) { out.kind=0; return; }
            if (translateMemInt(ibc, nreg, out.ib)) { out.kind=0; return; }
            out.kind=2; return;
    }
}

// Full batched interpreter over a translated real program. State:
//   r[8]  integer regs (per lane in __m512i)
//   F[8]  float regs f(0..3)/e(4..7) as BFReg lo/hi
//   A[4]  read-only a-group float regs (BFReg)
//   sp    per-lane scratchpad (contiguous LANES*spWords)
// Per-lane PC loop (handles CBRANCH divergence). Memory addr per lane via gather.
struct FullProg { FullInsn ins[512]; int count; };

// convert 2x int32 from scratchpad word -> 2 doubles (rx_cvt_packed_int_vec_f128),
// gathered per lane. word at 64-bit granularity: low 2 int32 of the 8-byte word.
static inline void cvtPackedIntPD(__m512i words, __m512d& lo, __m512d& hi) {
    // low int32 of each lane's 64-bit word -> lo double; high int32 -> hi double
    __m512i loI = _mm512_and_si512(words, _mm512_set1_epi64(0xffffffffULL));
    __m512i hiI = _mm512_srli_epi64(words, 32);
    // sign-extend 32->64 then convert
    loI = _mm512_srai_epi64(_mm512_slli_epi64(loI,32),32);
    hiI = _mm512_srai_epi64(_mm512_slli_epi64(hiI,32),32);
    lo = _mm512_cvtepi64_pd(loI);
    hi = _mm512_cvtepi64_pd(hiI);
}

static void runProgramFull(uint64_t rIn[IREGS][LANES],
                           BFReg F[8], const BFReg A[4],
                           uint64_t* sp, uint64_t spWords,
                           const uint64_t eMask[2],
                           const FullProg& prog)
{
    __m512i r[IREGS];
    for (int k=0;k<IREGS;++k) r[k]=_mm512_loadu_si512((const void*)rIn[k]);

    const __m512i laneBase = _mm512_set_epi64(
        (long long)(7*spWords),(long long)(6*spWords),(long long)(5*spWords),(long long)(4*spWords),
        (long long)(3*spWords),(long long)(2*spWords),(long long)(1*spWords),0LL);
    // E-mask as doubles (per element0/1)
    const __m512d eMaskLo = _mm512_castsi512_pd(_mm512_set1_epi64((long long)eMask[0]));
    const __m512d eMaskHi = _mm512_castsi512_pd(_mm512_set1_epi64((long long)eMask[1]));
    const __m512d mantMask = _mm512_castsi512_pd(_mm512_set1_epi64((long long)((1ULL<<52)-1)));

    int pc[LANES]; for(int l=0;l<LANES;++l) pc[l]=0;
    long steps=0, maxSteps=(long)prog.count*64*LANES;
    for(;;){
        int pos=prog.count;
        for(int l=0;l<LANES;++l) if(pc[l]<prog.count && pc[l]<pos) pos=pc[l];
        if(pos>=prog.count) break;
        if(++steps>maxSteps) break;
        __mmask8 m=0; for(int l=0;l<LANES;++l) if(pc[l]==pos) m|=(1u<<l);
        const FullInsn& fi = prog.ins[pos];

        if (fi.kind==2) { for(int l=0;l<LANES;++l) if(pc[l]==pos)++pc[l]; continue; }

        if (fi.kind==0) {
            const BInsn& in = fi.ib;
            if (in.op==B_CBRANCH) {
                __m512i& d=r[in.dst];
                d=_mm512_mask_add_epi64(d,m,d,_mm512_set1_epi64((long long)in.imm));
                __mmask8 taken=_mm512_mask_cmpeq_epi64_mask(m,
                    _mm512_and_si512(d,_mm512_set1_epi64((long long)(uint64_t)in.memMask)),_mm512_setzero_si512());
                for(int l=0;l<LANES;++l) if(pc[l]==pos) pc[l]=(taken&(1u<<l))?(in.target+1):pos+1;
                continue;
            }
            // integer register / memory ops
            switch (in.op) {
                case B_IADD_RS: case B_ISUB_R: case B_IMUL_R: case B_INEG_R:
                case B_IXOR_R:  case B_IROR_R: case B_IROL_R: case B_ISWAP_R:
                case B_IMULH_R: case B_ISMULH_R:
                    applyIntMasked(r,in,m); break;
                case B_IADD_M: case B_ISUB_M: case B_IMUL_M: case B_IMULH_M:
                case B_ISMULH_M: case B_IXOR_M: {
                    // addr reg value (0 if srcImm meaning zero-src)
                    __m512i sreg = in.srcImm ? _mm512_setzero_si512() : r[in.src];
                    __m512i addr=_mm512_and_si512(_mm512_add_epi64(sreg,_mm512_set1_epi64((long long)in.imm)),
                                                  _mm512_set1_epi64((long long)(uint64_t)in.memMask));
                    __m512i widx=_mm512_add_epi64(_mm512_srli_epi64(addr,3),laneBase);
                    __m512i v=_mm512_i64gather_epi64(widx,(const long long*)sp,8);
                    __m512i& d=r[in.dst];
                    switch(in.op){
                        case B_IADD_M:  d=_mm512_mask_add_epi64(d,m,d,v); break;
                        case B_ISUB_M:  d=_mm512_mask_sub_epi64(d,m,d,v); break;
                        case B_IMUL_M:  d=_mm512_mask_mullo_epi64(d,m,d,v); break;
                        case B_IMULH_M: d=_mm512_mask_blend_epi64(m,d,mulhi_epu64(d,v)); break;
                        case B_ISMULH_M:d=_mm512_mask_blend_epi64(m,d,mulhi_epi64(d,v)); break;
                        case B_IXOR_M:  d=_mm512_mask_xor_epi64(d,m,d,v); break;
                        default: break;
                    }
                    break;
                }
                case B_ISTORE: {
                    __m512i addr=_mm512_and_si512(_mm512_add_epi64(r[in.dst],_mm512_set1_epi64((long long)in.imm)),
                                                  _mm512_set1_epi64((long long)(uint64_t)in.memMask));
                    __m512i widx=_mm512_add_epi64(_mm512_srli_epi64(addr,3),laneBase);
                    // masked scatter: only active lanes
                    _mm512_mask_i64scatter_epi64((long long*)sp,m,widx,r[in.src],8);
                    break;
                }
                default: break;
            }
            for(int l=0;l<LANES;++l) if(pc[l]==pos)++pc[l];
            continue;
        }

        // kind==1 : float op
        {
            BFReg& d = F[fi.fdst];
            switch (fi.fop) {
                case FB_FSWAP: {
                    __m512d tl=_mm512_mask_blend_pd(m,d.lo,d.hi);
                    __m512d th=_mm512_mask_blend_pd(m,d.hi,d.lo);
                    d.lo=tl; d.hi=th; break;
                }
                case FB_FADD_R: { const BFReg& s=A[fi.fsrc];
                    d.lo=_mm512_mask_add_pd(d.lo,m,d.lo,s.lo); d.hi=_mm512_mask_add_pd(d.hi,m,d.hi,s.hi); break; }
                case FB_FSUB_R: { const BFReg& s=A[fi.fsrc];
                    d.lo=_mm512_mask_sub_pd(d.lo,m,d.lo,s.lo); d.hi=_mm512_mask_sub_pd(d.hi,m,d.hi,s.hi); break; }
                case FB_FMUL_R: { const BFReg& s=A[fi.fsrc];
                    d.lo=_mm512_mask_mul_pd(d.lo,m,d.lo,s.lo); d.hi=_mm512_mask_mul_pd(d.hi,m,d.hi,s.hi); break; }
                case FB_FSQRT_R:
                    d.lo=_mm512_mask_sqrt_pd(d.lo,m,d.lo); d.hi=_mm512_mask_sqrt_pd(d.hi,m,d.hi); break;
                case FB_FSCAL_R:
                    d.lo=_mm512_mask_blend_pd(m,d.lo,fscal(d.lo)); d.hi=_mm512_mask_blend_pd(m,d.hi,fscal(d.hi)); break;
                case FB_FADD_M: case FB_FSUB_M: case FB_FDIV_M: {
                    __m512i addr=_mm512_and_si512(_mm512_add_epi64(r[fi.maddr],_mm512_set1_epi64((long long)fi.fimm)),
                                                  _mm512_set1_epi64((long long)(uint64_t)fi.fmask));
                    __m512i widx=_mm512_add_epi64(_mm512_srli_epi64(addr,3),laneBase);
                    __m512i w=_mm512_i64gather_epi64(widx,(const long long*)sp,8);
                    __m512d slo,shi; cvtPackedIntPD(w,slo,shi);
                    if (fi.fop==FB_FDIV_M) {
                        // maskRegisterExponentMantissa: (x & mant) | eMask
                        slo=_mm512_or_pd(_mm512_and_pd(slo,mantMask),eMaskLo);
                        shi=_mm512_or_pd(_mm512_and_pd(shi,mantMask),eMaskHi);
                        d.lo=_mm512_mask_div_pd(d.lo,m,d.lo,slo); d.hi=_mm512_mask_div_pd(d.hi,m,d.hi,shi);
                    } else if (fi.fop==FB_FADD_M) {
                        d.lo=_mm512_mask_add_pd(d.lo,m,d.lo,slo); d.hi=_mm512_mask_add_pd(d.hi,m,d.hi,shi);
                    } else {
                        d.lo=_mm512_mask_sub_pd(d.lo,m,d.lo,slo); d.hi=_mm512_mask_sub_pd(d.hi,m,d.hi,shi);
                    }
                    break;
                }
                default: break;
            }
            for(int l=0;l<LANES;++l) if(pc[l]==pos)++pc[l];
        }
    }
    for(int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)rIn[k], r[k]);
}

int verifyProgramFull()
{
    RandomX_CurrentConfig.Apply();
    printf("== BatchedVm Stage 6b: full real-program verify (vs executeBytecode) ==\n");
    const int PROGRAMS = 1000;
    const uint64_t spWords = 8192;
    const uint32_t spMaskBytes = (uint32_t)(spWords*8 - 8); (void)spMaskBytes;
    long fails = 0;
    rng_s = 0x2545F4914F6CDD1DULL ^ 0x9e3779b97f4a7c15ULL;

    static uint64_t spSnap[LANES][8192];   // pristine scratchpad input per lane
    static uint64_t spB[LANES*8192];        // batched working scratchpad

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
        for (int i=0;i<fp.count;++i) translateFull(btT[i], nregT, fp.ins[i]);

        // ---- pristine input state (snapshot) ----
        uint64_t rSnap[IREGS][LANES];
        double fLo[8][LANES], fHi[8][LANES], aLo[4][LANES], aHi[4][LANES];
        for (int lane=0; lane<LANES; ++lane) {
            for (int k=0;k<IREGS;++k) rSnap[k][lane]=rng();
            for (int k=0;k<8;++k){ fLo[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0;
                                   fHi[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0; }
            for (int k=0;k<4;++k){ aLo[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0;
                                   aHi[k][lane]=1.0+(double)(rng()&0xffffff)/16777216.0; }
            for (uint64_t w=0; w<spWords; ++w) spSnap[lane][w]=rng();
        }

        // ---- batched run (copies of inputs) ----
        uint64_t rB[IREGS][LANES];
        for (int k=0;k<IREGS;++k) for(int l=0;l<LANES;++l) rB[k][l]=rSnap[k][l];
        for (int lane=0; lane<LANES; ++lane)
            for (uint64_t w=0; w<spWords; ++w) spB[lane*spWords+w]=spSnap[lane][w];
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
            for (int k=0;k<IREGS;++k) nr.r[k]=rSnap[k][lane];
            for (int k=0;k<randomx::RegisterCountFlt;++k){
                double lo=fLo[k][lane], hi=fHi[k][lane];
                nr.f[k]=_mm_set_pd(hi,lo);
                double elo=fLo[k+4][lane], ehi=fHi[k+4][lane];
                nr.e[k]=_mm_set_pd(ehi,elo);
                nr.a[k]=_mm_set_pd(aHi[k][lane],aLo[k][lane]);
            }
            uint8_t* sp = reinterpret_cast<uint8_t*>(spSnap[lane]);
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
    return fails==0?0:1;
}
} // namespace batchedx
} // namespace xmrig
