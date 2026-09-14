/* XMRig batched RandomX — reusable AVX-512 op primitives (header, inline).
 * No behaviour change: these are the same helpers previously local to BatchedVm.cpp,
 * exposed here so all batched translation units share them.
 */
#ifndef XMRIG_BATCHEDOPS_H
#define XMRIG_BATCHEDOPS_H

#include <cstdint>
#include <immintrin.h>
#include "crypto/batchedx/BatchedVm.h"

#if defined(__GNUC__)
#   pragma GCC push_options
#   pragma GCC target("avx512f,avx512dq,tune=znver4")
#endif

namespace xmrig { namespace batchedx {

// per-lane variable rotates
static inline __m512i rorv64(__m512i x, __m512i amt) {
    return _mm512_rorv_epi64(x, _mm512_and_si512(amt, _mm512_set1_epi64(63)));
}
static inline __m512i rolv64(__m512i x, __m512i amt) {
    return _mm512_rolv_epi64(x, _mm512_and_si512(amt, _mm512_set1_epi64(63)));
}

// 64x64 -> high 64 (unsigned) across 8 lanes, schoolbook from 32x32 products
static inline __m512i mulhi_epu64(__m512i a, __m512i b) {
    const __m512i lo32 = _mm512_set1_epi64(0xffffffffULL);
    __m512i al = _mm512_and_si512(a, lo32);
    __m512i ah = _mm512_srli_epi64(a, 32);
    __m512i bl = _mm512_and_si512(b, lo32);
    __m512i bh = _mm512_srli_epi64(b, 32);
    __m512i ll = _mm512_mul_epu32(al, bl);
    __m512i lh = _mm512_mul_epu32(al, bh);
    __m512i hl = _mm512_mul_epu32(ah, bl);
    __m512i hh = _mm512_mul_epu32(ah, bh);
    __m512i cross = _mm512_add_epi64(_mm512_srli_epi64(ll, 32),
                    _mm512_add_epi64(_mm512_and_si512(lh, lo32),
                                     _mm512_and_si512(hl, lo32)));
    __m512i high = _mm512_add_epi64(hh,
                   _mm512_add_epi64(_mm512_srli_epi64(lh, 32),
                   _mm512_add_epi64(_mm512_srli_epi64(hl, 32),
                                    _mm512_srli_epi64(cross, 32))));
    return high;
}
// signed high-mul: umulh(a,b) - (a<0?b:0) - (b<0?a:0)
static inline __m512i mulhi_epi64(__m512i a, __m512i b) {
    __m512i u = mulhi_epu64(a, b);
    const __m512i zero = _mm512_setzero_si512();
    __mmask8 an = _mm512_cmplt_epi64_mask(a, zero);
    __mmask8 bn = _mm512_cmplt_epi64_mask(b, zero);
    u = _mm512_mask_sub_epi64(u, an, u, b);
    u = _mm512_mask_sub_epi64(u, bn, u, a);
    return u;
}

// float masks / helpers (E-register domain + FSCAL)
static const uint64_t FSCAL_MASK     = 0x80F0000000000000ULL;
static const uint64_t E_MANTISSA_MASK = (1ULL << 56) - 1;
static const uint64_t E_EXP_MASK      = 0x3000000000000000ULL;

static inline __m512d emask(__m512d v) {
    __m512i b = _mm512_castpd_si512(v);
    b = _mm512_and_si512(b, _mm512_set1_epi64((long long)E_MANTISSA_MASK));
    b = _mm512_or_si512 (b, _mm512_set1_epi64((long long)E_EXP_MASK));
    return _mm512_castsi512_pd(b);
}
static inline __m512d fscal(__m512d v) {
    const __m512i m = _mm512_set1_epi64((long long)FSCAL_MASK);
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v), m));
}

// float register across 8 lanes: two doubles per lane -> lo/hi vectors
struct BFReg { __m512d lo, hi; };

} } // namespace xmrig::batchedx

#if defined(__GNUC__)
#   pragma GCC pop_options
#endif

#endif
