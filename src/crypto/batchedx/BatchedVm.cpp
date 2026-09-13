/* XMRig — batched AVX-512 RandomX (8-lane), implementation.
 *
 * Stage 1: 8-lane interpreter for the RandomX integer register ops, plus a bit-exact
 * self-test against scalar semantics. Compiled in only when WITH_BATCHEDX=ON.
 */

#include "crypto/batchedx/BatchedVm.h"

#include <immintrin.h>
#include <cstdio>
#include <cstring>

#if defined(__GNUC__)
#   pragma GCC target("avx512f,avx512dq")
#endif

namespace xmrig {
namespace batchedx {


// ---- AVX-512 helpers ----------------------------------------------------------

// per-lane variable rotate-right by (amt & 63): AVX-512 has _mm512_rorv_epi64
static inline __m512i rorv64(__m512i x, __m512i amt) {
  return _mm512_rorv_epi64(x, _mm512_and_si512(amt, _mm512_set1_epi64(63)));
}
static inline __m512i rolv64(__m512i x, __m512i amt) {
  return _mm512_rolv_epi64(x, _mm512_and_si512(amt, _mm512_set1_epi64(63)));
}


// 64x64 -> high 64 across 8 lanes. AVX-512 has no 64-bit high-mul; synthesize from
// 32x32 partial products (schoolbook), then correct for signedness if needed.
static inline __m512i mulhi_epu64(__m512i a, __m512i b) {
    const __m512i lo32 = _mm512_set1_epi64(0xffffffffULL);
    __m512i al = _mm512_and_si512(a, lo32);
    __m512i ah = _mm512_srli_epi64(a, 32);
    __m512i bl = _mm512_and_si512(b, lo32);
    __m512i bh = _mm512_srli_epi64(b, 32);

    __m512i ll = _mm512_mul_epu32(al, bl);          // al*bl
    __m512i lh = _mm512_mul_epu32(al, bh);          // al*bh
    __m512i hl = _mm512_mul_epu32(ah, bl);          // ah*bl
    __m512i hh = _mm512_mul_epu32(ah, bh);          // ah*bh

    // cross = (ll>>32) + (lh & lo32) + (hl & lo32)
    __m512i cross = _mm512_add_epi64(_mm512_srli_epi64(ll, 32),
                    _mm512_add_epi64(_mm512_and_si512(lh, lo32),
                                     _mm512_and_si512(hl, lo32)));
    // high = hh + (lh>>32) + (hl>>32) + (cross>>32)
    __m512i high = _mm512_add_epi64(hh,
                   _mm512_add_epi64(_mm512_srli_epi64(lh, 32),
                   _mm512_add_epi64(_mm512_srli_epi64(hl, 32),
                                    _mm512_srli_epi64(cross, 32))));
    return high;
}

// signed high-mul: umulh(a,b) - (a<0 ? b : 0) - (b<0 ? a : 0)
static inline __m512i mulhi_epi64(__m512i a, __m512i b) {
    __m512i u = mulhi_epu64(a, b);
    const __m512i zero = _mm512_setzero_si512();
    __mmask8 an = _mm512_cmplt_epi64_mask(a, zero);
    __mmask8 bn = _mm512_cmplt_epi64_mask(b, zero);
    u = _mm512_mask_sub_epi64(u, an, u, b);
    u = _mm512_mask_sub_epi64(u, bn, u, a);
    return u;
}

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
    }
  }
}


// ---- self-test ---------------------------------------------------------------

static uint64_t rng_s;
static inline uint64_t rng() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }

int verifyIntegerOps(){
  printf("== BatchedVm Stage 1: integer ops verify ==\n");
  rng_s = 0x9e3779b97f4a7c15ULL;

  const int PROGRAMS = 20000;
  const int PROGLEN  = 256;
  int fails = 0;

  for (int p = 0; p < PROGRAMS; ++p) {
    // random program (integer register ops only)
    BInsn prog[PROGLEN];
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
static const uint64_t FSCAL_MASK = 0x80F0000000000000ULL;

// E-register exponent/mantissa mask (RandomX keeps E-regs positive & bounded).
static const uint64_t E_MANTISSA_MASK = (1ULL << 56) - 1;      // dynamicMantissaMask
static const uint64_t E_EXP_MASK      = 0x3000000000000000ULL; // constExponentBits(0x300)<<52

static inline __m512d emask(__m512d v) {
    __m512i b = _mm512_castpd_si512(v);
    b = _mm512_and_si512(b, _mm512_set1_epi64((long long)E_MANTISSA_MASK));
    b = _mm512_or_si512 (b, _mm512_set1_epi64((long long)E_EXP_MASK));
    return _mm512_castsi512_pd(b);
}

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
struct BFReg { __m512d lo, hi; };

static inline __m512d fscal(__m512d v) {
    const __m512i m = _mm512_set1_epi64((long long)FSCAL_MASK);
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v), m));
}

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


} // namespace batchedx
} // namespace xmrig
