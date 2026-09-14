/* XMRig — batched AVX-512 RandomX: scalar reference oracle. */
#include "crypto/batchedx/BatchedInternal.h"

namespace xmrig {
namespace batchedx {

static inline double emask_s(double v) {
    uint64_t b; memcpy(&b, &v, 8);
    b = (b & E_MANTISSA_MASK) | E_EXP_MASK;
    double r; memcpy(&r, &b, 8); return r;
}

void scalarIntegerProgram(uint64_t regs[IREGS], const BInsn* prog, int count){
  for (int i = 0; i < count; ++i) {
    const BInsn& in = prog[i];
    const uint64_t s = regs[in.src];
    applyScalarIntOp(regs, in, s);
  }
}

void scalarMemoryProgram(uint64_t regs[IREGS], const BInsn* prog, int count, uint64_t* sp)
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

// scalar reference: real pc with jumps
void scalarBranchProgram(uint64_t regs[IREGS], const BInsn* prog, int count)
{
    long steps = 0, maxSteps = (long)count * 64;
    for (int pc = 0; pc < count; ) {
        if (++steps > maxSteps) break;
        const BInsn& in = prog[pc];
        uint64_t& d = regs[in.dst];
        const uint64_t s = regs[in.src];
        switch (in.op) {
            case B_CBRANCH: d += in.imm; if ((d & (uint64_t)in.memMask) == 0) pc = in.target; else ++pc; break;
            default: applyScalarIntOp(regs, in, s); ++pc; break;
        }
    }
}

// scalar reference: each register is 2 doubles [0]=lo [1]=hi
void scalarFloatProgram(double reg[FREGS][2], const FInsn* prog, int count)
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

// scalar reference for the same restricted (int/branch) program on one lane.
void scalarProgramIntBranch(uint64_t regs[IREGS], const BProg& prog)
{
    long steps=0, maxSteps=(long)prog.count*64;
    for (int pc=0; pc<prog.count; ) {
        if (++steps>maxSteps) break;
        const BInsn& in=prog.ins[pc];
        if (!prog.ok[pc]) { ++pc; continue; }         // float/mem => no-op
        uint64_t& d=regs[in.dst]; const uint64_t s = in.srcImm ? in.srcVal : regs[in.src];
        switch (in.op) {
            case B_CBRANCH: d+=in.imm; if((d&(uint64_t)in.memMask)==0) pc=in.target+1; else ++pc; break;
            default: applyScalarIntOp(regs, in, s); ++pc; break;
        }
    }
}

} // namespace batchedx
} // namespace xmrig
