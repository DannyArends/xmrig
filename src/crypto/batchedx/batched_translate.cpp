/* XMRig - batched AVX-512 RandomX: bytecode -> BInsn/FullInsn translation
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

#include "crypto/batchedx/BatchedInternal.h"

namespace xmrig {
namespace batchedx {

// translate one compiled InstructionByteCode -> BInsn (integer + branch).
bool translateInt(const randomx::InstructionByteCode& ibc,
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

// Extend translateInt-style mapping for memory-integer ops. Returns true if mapped.
bool translateMemInt(const randomx::InstructionByteCode& ibc,
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

void translateFull(const randomx::InstructionByteCode& ibc,
                          const randomx::NativeRegisterFile& nreg, FullInsn& out, bool cfround)
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
        case IT::CFROUND:
            if (cfround) { out.kind=3; out.maddr=(uint8_t)(ibc.isrc-&nreg.r[0]); out.fimm=ibc.imm; }
            else          out.kind=2;
            return;
        case IT::NOP:     out.kind=2; return;
        default:
            if (translateInt(ibc, nreg, out.ib)) { out.kind=0; return; }
            if (translateMemInt(ibc, nreg, out.ib)) { out.kind=0; return; }
            out.kind=2; return;
    }
}

} // namespace batchedx
} // namespace xmrig
