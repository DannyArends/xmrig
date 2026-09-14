/* XMRig — batched AVX-512 RandomX: batched SIMD execution. */
#include "crypto/batchedx/BatchedInternal.h"

namespace xmrig {
namespace batchedx {

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
        d = _mm512_add_epi64(d, _mm512_add_epi64(_mm512_slli_epi64(s, in.shift), _mm512_set1_epi64((long long)in.imm)));
        break;
      case B_ISUB_R:  // dst -= src
        d = _mm512_sub_epi64(d, s);
        break;
      case B_IMUL_R:  // dst *= src   (low 64 bits)
        d = _mm512_mullo_epi64(d, s);
        break;
      case B_INEG_R:  // dst = ~dst + 1
        d = _mm512_add_epi64(_mm512_xor_si512(d, _mm512_set1_epi64(-1)), _mm512_set1_epi64(1));
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

// batched program with CBRANCH. Uses per-lane pc.
void runBranchProgram(uint64_t regs[IREGS][LANES], const BInsn* prog, int count)
{
    __m512i r[IREGS];
    for (int k = 0; k < IREGS; ++k) r[k] = _mm512_loadu_si512((const void*)regs[k]);

    int pc[LANES];
    for (int l = 0; l < LANES; ++l) pc[l] = 0;

    // Per-lane step budget matching scalarBranchProgram. The synthetic generator can
    // emit non-terminating loops real RandomX never produces; a global cap would give a
    // stuck lane the whole budget (starving others), diverging from the per-lane scalar.
    long lsteps[LANES]; for (int l = 0; l < LANES; ++l) lsteps[l] = 0;
    const long cap = (long)count * 64;

    for (;;) {
        // find min active pc
        int pos = count;
        for (int l = 0; l < LANES; ++l) if (pc[l] < count && pc[l] < pos) pos = pc[l];
        if (pos >= count) break;                 // all lanes done

        // active mask = lanes at this position, retiring any lane past its budget
        __mmask8 m = 0;
        for (int l = 0; l < LANES; ++l) if (pc[l] == pos) {
            if (++lsteps[l] > cap) pc[l] = count;   // lane exhausted -> done (matches scalar break)
            else                   m |= (1u << l);
        }
        if (!m) continue;                           // every lane here was exhausted

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

void runFloatProgram(BFReg reg[FREGS], const FInsn* prog, int count)
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

void runProgramIntBranch(uint64_t regs[IREGS][LANES], const BProg& prog)
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

// Interpret one translated program over the given register vectors, in place.
// No scratchpad framing; the program's own memory ops address sp via laneBase.
void runBytecodeVec(__m512i r[IREGS], BFReg F[8], const BFReg A[4],
                    uint64_t* sp, uint64_t spWords,
                    const uint64_t eMask[2], const FullProg& prog, int rmode[LANES])
{
    const __m512i laneBase = _mm512_set_epi64(
        (long long)(7*spWords),(long long)(6*spWords),(long long)(5*spWords),(long long)(4*spWords),
        (long long)(3*spWords),(long long)(2*spWords),(long long)(1*spWords),0LL);
    // E-mask as doubles (per element0/1)
    const __m512d eMaskLo = _mm512_castsi512_pd(_mm512_set1_epi64((long long)eMask[0]));
    const __m512d eMaskHi = _mm512_castsi512_pd(_mm512_set1_epi64((long long)eMask[1]));
    const __m512d mantMask = _mm512_castsi512_pd(_mm512_set1_epi64((long long)E_MANTISSA_MASK));

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

        if (fi.kind==3) {                                   // CFROUND: per-lane rounding-mode update
            uint64_t rv[LANES]; _mm512_storeu_si512(rv, r[fi.maddr]);
            for(int l=0;l<LANES;++l) if(pc[l]==pos){
                uint64_t v=s_rotr(rv[l],(unsigned)fi.fimm);
                if(!RandomX_CurrentConfig.Tweak_V2_CFROUND || (v&60)==0) rmode[l]=(int)(v&3);  // match exe_CFROUND gate
                ++pc[l];
            }
            continue;
        }

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
            if (fi.fop==FB_FSWAP) {                          // rounding-independent
                __m512d tl=_mm512_mask_blend_pd(m,d.lo,d.hi);
                __m512d th=_mm512_mask_blend_pd(m,d.hi,d.lo);
                d.lo=tl; d.hi=th;
            } else if (fi.fop==FB_FSCAL_R) {                 // rounding-independent (bit flip)
                d.lo=_mm512_mask_blend_pd(m,d.lo,fscal(d.lo)); d.hi=_mm512_mask_blend_pd(m,d.hi,fscal(d.hi));
            } else {
                // rounding-sensitive: resolve operand once, then apply per-lane mode in <=4 masked passes
                __m512d slo=_mm512_setzero_pd(), shi=_mm512_setzero_pd();
                if (fi.fop==FB_FADD_R||fi.fop==FB_FSUB_R||fi.fop==FB_FMUL_R) {
                    const BFReg& s=A[fi.fsrc]; slo=s.lo; shi=s.hi;
                } else if (fi.fop==FB_FADD_M||fi.fop==FB_FSUB_M||fi.fop==FB_FDIV_M) {
                    __m512i addr=_mm512_and_si512(_mm512_add_epi64(r[fi.maddr],_mm512_set1_epi64((long long)fi.fimm)),
                                                  _mm512_set1_epi64((long long)(uint64_t)fi.fmask));
                    __m512i widx=_mm512_add_epi64(_mm512_srli_epi64(addr,3),laneBase);
                    __m512i w=_mm512_i64gather_epi64(widx,(const long long*)sp,8);
                    cvtPackedIntPD(w,slo,shi);
                    if (fi.fop==FB_FDIV_M) {                 // maskRegisterExponentMantissa: (x & mant) | eMask
                        slo=_mm512_or_pd(_mm512_and_pd(slo,mantMask),eMaskLo);
                        shi=_mm512_or_pd(_mm512_and_pd(shi,mantMask),eMaskHi);
                    }
                }
                __mmask8 sub[4]={0,0,0,0};
                for(int l=0;l<LANES;++l) if(pc[l]==pos) sub[rmode[l]]|=(__mmask8)(1u<<l);
                for(int md=0;md<4;++md){
                    if(!sub[md]) continue;
                    _mm_setcsr(0x9FC0u|((unsigned)md<<13));
                    __mmask8 sm=sub[md];
                    switch(fi.fop){
                        case FB_FADD_R: case FB_FADD_M:
                            d.lo=_mm512_mask_add_pd(d.lo,sm,d.lo,slo); d.hi=_mm512_mask_add_pd(d.hi,sm,d.hi,shi); break;
                        case FB_FSUB_R: case FB_FSUB_M:
                            d.lo=_mm512_mask_sub_pd(d.lo,sm,d.lo,slo); d.hi=_mm512_mask_sub_pd(d.hi,sm,d.hi,shi); break;
                        case FB_FMUL_R:
                            d.lo=_mm512_mask_mul_pd(d.lo,sm,d.lo,slo); d.hi=_mm512_mask_mul_pd(d.hi,sm,d.hi,shi); break;
                        case FB_FDIV_M:
                            d.lo=_mm512_mask_div_pd(d.lo,sm,d.lo,slo); d.hi=_mm512_mask_div_pd(d.hi,sm,d.hi,shi); break;
                        case FB_FSQRT_R:
                            d.lo=_mm512_mask_sqrt_pd(d.lo,sm,d.lo); d.hi=_mm512_mask_sqrt_pd(d.hi,sm,d.hi); break;
                        default: break;
                    }
                }
            }
            for(int l=0;l<LANES;++l) if(pc[l]==pos)++pc[l];
        }
    }
}

// Thin wrapper: load int regs, interpret once, store back (Stage 6 entry point).
void runProgramFull(uint64_t rIn[IREGS][LANES], BFReg F[8], const BFReg A[4],
                    uint64_t* sp, uint64_t spWords, const uint64_t eMask[2], const FullProg& prog)
{
    __m512i r[IREGS]; for (int k=0;k<IREGS;++k) r[k]=_mm512_loadu_si512((const void*)rIn[k]);
    int rmode[LANES]; for(int l=0;l<LANES;++l) rmode[l]=0;
    runBytecodeVec(r, F, A, sp, spWords, eMask, prog, rmode);
    for (int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)rIn[k], r[k]);
}

// 8-wide RandomX per-program execution loop (mirror of InterpretedVm::execute).
// r/F carry in the register file (F[0..3]=f, F[4..7]=e) and carry out the result;
// ma/mx carry the memory registers; rounding mode persists across iterations.
void runBatchedExecute(__m512i r[IREGS], BFReg F[8], const BFReg A[4],
                       uint64_t* sp, uint64_t spWords,
                       const uint64_t eMask[2], const FullProg& prog,
                       __m512i& ma, __m512i& mx,
                       int rr0, int rr1, int rr2, int rr3,
                       uint64_t datasetOffset, const uint64_t* dataset, uint64_t dmask,
                       uint32_t iterations, int rmode[LANES])
{
    const __m512i laneBase = _mm512_set_epi64(
        (long long)(7*spWords),(long long)(6*spWords),(long long)(5*spWords),(long long)(4*spWords),
        (long long)(3*spWords),(long long)(2*spWords),(long long)(1*spWords),0LL);
    const __m512d eMaskLo  = _mm512_castsi512_pd(_mm512_set1_epi64((long long)eMask[0]));
    const __m512d eMaskHi  = _mm512_castsi512_pd(_mm512_set1_epi64((long long)eMask[1]));
    const __m512d mantMask = _mm512_castsi512_pd(_mm512_set1_epi64((long long)E_MANTISSA_MASK));
    const __m512i sp3mask  = _mm512_set1_epi64((long long)(uint32_t)ScratchpadL3Mask64);
    const __m512i clmask   = _mm512_set1_epi64((long long)(uint64_t)CacheLineAlignMask);
    const __m512i dmaskv   = _mm512_set1_epi64((long long)dmask);
    const __m512i lo32     = _mm512_set1_epi64(0xFFFFFFFFll);
    const bool prefetchTweak = RandomX_CurrentConfig.Tweak_V2_PREFETCH;

    __m512i spAddr0 = mx;
    __m512i spAddr1 = ma;

    for(uint32_t ic=0; ic<iterations; ++ic){
        __m512i spMix = _mm512_xor_si512(r[rr0], r[rr1]);
        spAddr0 = _mm512_and_si512(_mm512_xor_si512(spAddr0, spMix), sp3mask);
        spAddr1 = _mm512_and_si512(_mm512_xor_si512(spAddr1, _mm512_srli_epi64(spMix,32)), sp3mask);

        const __m512i w0 = _mm512_add_epi64(laneBase, _mm512_srli_epi64(spAddr0,3));
        const __m512i w1 = _mm512_add_epi64(laneBase, _mm512_srli_epi64(spAddr1,3));

        for(int i=0;i<IREGS;++i){
            __m512i idx=_mm512_add_epi64(w0,_mm512_set1_epi64(i));
            r[i]=_mm512_xor_si512(r[i], _mm512_i64gather_epi64(idx,(const long long*)sp,8));
        }
        for(int i=0;i<randomx::RegisterCountFlt;++i){
            __m512i idx=_mm512_add_epi64(w1,_mm512_set1_epi64(i));
            __m512i wq=_mm512_i64gather_epi64(idx,(const long long*)sp,8);
            cvtPackedIntPD(wq, F[i].lo, F[i].hi);
        }
        for(int i=0;i<randomx::RegisterCountFlt;++i){
            __m512i idx=_mm512_add_epi64(w1,_mm512_set1_epi64(randomx::RegisterCountFlt+i));
            __m512i wq=_mm512_i64gather_epi64(idx,(const long long*)sp,8);
            __m512d elo,ehi; cvtPackedIntPD(wq,elo,ehi);
            F[randomx::RegisterCountFlt+i].lo=_mm512_or_pd(_mm512_and_pd(elo,mantMask),eMaskLo);
            F[randomx::RegisterCountFlt+i].hi=_mm512_or_pd(_mm512_and_pd(ehi,mantMask),eMaskHi);
        }

        runBytecodeVec(r, F, A, sp, spWords, eMask, prog, rmode);

        __m512i readPtr=_mm512_add_epi64(_mm512_set1_epi64((long long)datasetOffset),
                                         _mm512_and_si512(ma, clmask));
        __m512i mpx=_mm512_and_si512(_mm512_xor_si512(r[rr2],r[rr3]), lo32);
        if(prefetchTweak) ma=_mm512_and_si512(_mm512_xor_si512(ma,mpx), lo32);
        else              mx=_mm512_and_si512(_mm512_xor_si512(mx,mpx), lo32);

        __m512i dw=_mm512_srli_epi64(readPtr,3);
        for(int i=0;i<IREGS;++i){
            __m512i idx=_mm512_and_si512(_mm512_add_epi64(dw,_mm512_set1_epi64(i)), dmaskv);
            r[i]=_mm512_xor_si512(r[i], _mm512_i64gather_epi64(idx,(const long long*)dataset,8));
        }

        { __m512i t=mx; mx=ma; ma=t; }

        for(int i=0;i<IREGS;++i){
            __m512i idx=_mm512_add_epi64(w1,_mm512_set1_epi64(i));
            _mm512_i64scatter_epi64((long long*)sp, idx, r[i], 8);
        }
        for(int i=0;i<randomx::RegisterCountFlt;++i){
            F[i].lo=_mm512_xor_pd(F[i].lo,F[randomx::RegisterCountFlt+i].lo);
            F[i].hi=_mm512_xor_pd(F[i].hi,F[randomx::RegisterCountFlt+i].hi);
            _mm512_i64scatter_epi64((long long*)sp, _mm512_add_epi64(w0,_mm512_set1_epi64(2*i)),
                                    _mm512_castpd_si512(F[i].lo), 8);
            _mm512_i64scatter_epi64((long long*)sp, _mm512_add_epi64(w0,_mm512_set1_epi64(2*i+1)),
                                    _mm512_castpd_si512(F[i].hi), 8);
        }

        spAddr0=_mm512_setzero_si512();
        spAddr1=_mm512_setzero_si512();
    }
}

} // namespace batchedx
} // namespace xmrig
