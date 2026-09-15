/* XMRig - batched AVX-512 RandomX: internal shared declarations
 * Types, inline helpers and cross-TU entry points shared by the batchedx
 * translation units (translate / exec / ref / verify). Not a public API.
 *
 * Copyright 2018-2020 SChernykh    <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig        <https://github.com/xmrig>, <support@xmrig.com>
 * Copyright (c) 2026  Danny Arends <https://github.com/DannyArends>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef XMRIG_BATCHEDINTERNAL_H
#define XMRIG_BATCHEDINTERNAL_H

#include "crypto/batchedx/BatchedVm.h"
#include "crypto/batchedx/BatchedOps.h"

#include "crypto/randomx/bytecode_machine.hpp"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/aes_hash.hpp"

namespace xmrig {
namespace batchedx {

static constexpr int FREGS = 4;   // RandomX f-group registers

// RandomX default MXCSR: flush-to-zero, denormals-are-zero, round-to-nearest, exceptions off.
static constexpr unsigned kMxcsrDefault = 0x9FC0u;

// ---- float op set (Stage 3 / float verify) ----
enum FOp : uint8_t { F_FSWAP_R, F_FADD_R, F_FSUB_R, F_FMUL_R, F_FSQRT_R, F_FSCAL_R };
struct FInsn { FOp op; uint8_t dst; uint8_t src; };   // dst/src index 0..FREGS-1

// ---- translated-program containers ----
struct BProg { BInsn ins[512]; bool ok[512]; int count; };
enum FBOp : uint8_t { FB_NONE, FB_FSWAP, FB_FADD_R, FB_FSUB_R, FB_FMUL_R, FB_FSQRT_R,
                      FB_FSCAL_R, FB_FADD_M, FB_FSUB_M, FB_FDIV_M };

struct FullInsn {
    uint8_t kind;   // 0=int/branch(ib), 1=float(fop), 2=skip, 3=cfround
    BInsn   ib;
    FBOp    fop;
    uint8_t fdst;   // 0..7 (f=0..3, e=4..7)
    uint8_t fsrc;   // a-group 0..3
    uint8_t maddr;  // r-reg for float-M address
    uint64_t fimm;
    uint32_t fmask;
};
struct FullProg { FullInsn ins[512]; int count; };

// ---- shared scalar / masked helpers (inline, one copy per TU) ----
#if defined(__GNUC__)
#   pragma GCC push_options
#   pragma GCC target("avx512f,avx512dq,tune=native")
#endif
static inline uint64_t s_rotr(uint64_t x, unsigned c) { c &= 63; return c ? (x >> c) | (x << (64 - c)) : x; }
static inline uint64_t s_rotl(uint64_t x, unsigned c) { c &= 63; return c ? (x << c) | (x >> (64 - c)) : x; }

/** Word-index of each lane's scratchpad region: lane l starts at l*spWords. */
static inline __m512i makeLaneBase(uint64_t spWords) {
    return _mm512_set_epi64((long long)(7*spWords), (long long)(6*spWords), (long long)(5*spWords), (long long)(4*spWords),
                            (long long)(3*spWords), (long long)(2*spWords), (long long)(1*spWords), 0LL);
}

/** Set MXCSR to the RandomX default with rounding mode (0..3) applied. */
static inline void setRoundingMode(int mode) { _mm_setcsr(kMxcsrDefault | ((unsigned)mode << 13)); }

/** Scalar reference for one register-only integer op (IADD_RS..ISMULH_R, ISWAP_R). */
static inline void applyScalarIntOp(uint64_t regs[IREGS], const BInsn& in, uint64_t s) {
    uint64_t& d = regs[in.dst];
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

#if defined(__GNUC__)
#   pragma GCC pop_options
#endif

// ---- cross-TU entry points ----
void runMemoryProgram(uint64_t regs[IREGS][LANES], const BInsn* prog, int count, uint64_t* sp, uint64_t spWords);
void runBranchProgram(uint64_t regs[IREGS][LANES], const BInsn* prog, int count);
void runFloatProgram(BFReg reg[FREGS], const FInsn* prog, int count);
void runProgramIntBranch(uint64_t regs[IREGS][LANES], const BProg& prog);
void runProgramFull(uint64_t rIn[IREGS][LANES], BFReg F[8], const BFReg A[4], uint64_t* sp, uint64_t spWords, const uint64_t eMask[2], const FullProg& prog);
void runBytecodeVecPerLane(__m512i r[IREGS], BFReg F[8], const BFReg A[4], uint64_t* sp, uint64_t spWords, __m512d eMaskLo, __m512d eMaskHi, const FullProg* prog, int rmode[LANES]);
void runBatchedExecutePerLane(__m512i r[IREGS], BFReg F[8], const BFReg A[4], uint64_t* sp, uint64_t spWords, __m512d eMaskLo, __m512d eMaskHi, const FullProg* prog, __m512i& ma, __m512i& mx, __m512i rr0, __m512i rr1, __m512i rr2, __m512i rr3, __m512i datasetOffset, const uint64_t* dataset, uint64_t dmask, uint32_t iterations, int rmode[LANES]);
void runBatchedExecute(__m512i r[IREGS], BFReg F[8], const BFReg A[4], uint64_t* sp, uint64_t spWords, const uint64_t eMask[2], const FullProg& prog, __m512i& ma, __m512i& mx, int rr0, int rr1, int rr2, int rr3, uint64_t datasetOffset, const uint64_t* dataset, uint64_t dmask, uint32_t iterations, int rmode[LANES]);
void scalarIntegerProgram(uint64_t regs[IREGS], const BInsn* prog, int count);
void scalarMemoryProgram(uint64_t regs[IREGS], const BInsn* prog, int count, uint64_t* sp);
void scalarBranchProgram(uint64_t regs[IREGS], const BInsn* prog, int count);
void scalarFloatProgram(double reg[FREGS][2], const FInsn* prog, int count);
void scalarProgramIntBranch(uint64_t regs[IREGS], const BProg& prog);
bool translateInt(const randomx::InstructionByteCode& ibc, const randomx::NativeRegisterFile& nreg, BInsn& out);
bool translateMemInt(const randomx::InstructionByteCode& ibc, const randomx::NativeRegisterFile& nreg, BInsn& out);
void translateFull(const randomx::InstructionByteCode& ibc, const randomx::NativeRegisterFile& nreg, FullInsn& out, bool cfround);

} // namespace batchedx
} // namespace xmrig

#endif // XMRIG_BATCHEDINTERNAL_H
