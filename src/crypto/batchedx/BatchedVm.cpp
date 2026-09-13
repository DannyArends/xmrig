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


} // namespace batchedx
} // namespace xmrig