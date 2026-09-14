/* XMRig — batched AVX-512 RandomX: batched SIMD execution. */
#include "crypto/batchedx/BatchedInternal.h"

namespace xmrig {
namespace batchedx {

#if defined(__GNUC__)
#   pragma GCC push_options
#   pragma GCC target("avx512f,avx512dq")
#endif

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

// ---- per-lane programs (increment 1: integer ops, linear, no branch) ----
// Each lane runs its OWN program. selectReg gathers r[idx[lane]] across the 8
// register vectors; scatterReg writes it back. Opcode-grouped: one masked pass
// per opcode present among the 8 lanes at this position.
static inline __m512i selectReg(const __m512i r[IREGS], __m512i idx)
{
    __m512i sel = r[0];
    for (int k=1;k<IREGS;++k)
        sel = _mm512_mask_mov_epi64(sel, _mm512_cmpeq_epi64_mask(idx, _mm512_set1_epi64(k)), r[k]);
    return sel;
}
static inline void scatterReg(__m512i r[IREGS], __m512i idx, __m512i val, __mmask8 active)
{
    for (int k=0;k<IREGS;++k)
        r[k] = _mm512_mask_mov_epi64(r[k], active & _mm512_cmpeq_epi64_mask(idx, _mm512_set1_epi64(k)), val);
}

// prog is [LANES][count] row-major: lane l instruction i at prog[l*count + i].
void runPerLaneInt(uint64_t regs[IREGS][LANES], const BInsn* prog, int count)
{
    __m512i r[IREGS];
    for (int k=0;k<IREGS;++k) r[k]=_mm512_loadu_si512((const void*)regs[k]);

    for (int pos=0; pos<count; ++pos) {
        alignas(64) long long ov[LANES],dv[LANES],sv[LANES],hv[LANES],iv[LANES];
        for (int l=0;l<LANES;++l){ const BInsn& ib=prog[(size_t)l*count+pos];
            ov[l]=ib.op; dv[l]=ib.dst; sv[l]=ib.src; hv[l]=ib.shift; iv[l]=(long long)ib.imm; }
        __m512i op=_mm512_loadu_si512(ov), dst=_mm512_loadu_si512(dv), src=_mm512_loadu_si512(sv),
                sh=_mm512_loadu_si512(hv), im=_mm512_loadu_si512(iv);

        __m512i s = selectReg(r, src);
        __m512i d = selectReg(r, dst);
        __m512i nd = d;
        auto OP=[&](BOp o){ return _mm512_cmpeq_epi64_mask(op, _mm512_set1_epi64((long long)o)); };
        nd=_mm512_mask_mov_epi64(nd, OP(B_IADD_RS), _mm512_add_epi64(d,_mm512_add_epi64(_mm512_sllv_epi64(s,sh),im)));
        nd=_mm512_mask_mov_epi64(nd, OP(B_ISUB_R),  _mm512_sub_epi64(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IMUL_R),  _mm512_mullo_epi64(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_INEG_R),  _mm512_add_epi64(_mm512_xor_si512(d,_mm512_set1_epi64(-1)),_mm512_set1_epi64(1)));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IXOR_R),  _mm512_xor_si512(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IROR_R),  _mm512_rorv_epi64(d,_mm512_and_si512(s,_mm512_set1_epi64(63))));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IROL_R),  _mm512_rolv_epi64(d,_mm512_and_si512(s,_mm512_set1_epi64(63))));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IMULH_R), mulhi_epu64(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_ISMULH_R),mulhi_epi64(d,s));

        // ISWAP: dst<-src, src<-dst  (only when dst!=src)
        __mmask8 swm = OP(B_ISWAP_R) & _mm512_cmpneq_epi64_mask(dst, src);
        nd = _mm512_mask_mov_epi64(nd, swm, s);   // value written to dst for swap lanes
        scatterReg(r, dst, nd, (__mmask8)0xFF);
        scatterReg(r, src, d, swm);
    }

    for (int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)regs[k], r[k]);
}

// increment 2: per-lane programs with per-lane pc + CBRANCH (control-flow divergence).
void runPerLaneIntBranch(uint64_t regs[IREGS][LANES], const BInsn* prog, int count)
{
    __m512i r[IREGS];
    for (int k=0;k<IREGS;++k) r[k]=_mm512_loadu_si512((const void*)regs[k]);

    int  pc[LANES];   for(int l=0;l<LANES;++l) pc[l]=0;
    long steps[LANES];for(int l=0;l<LANES;++l) steps[l]=0;
    const long cap=(long)count*64;

    for(;;){
        __mmask8 active=0;
        for(int l=0;l<LANES;++l) if(pc[l]>=0 && pc[l]<count && steps[l]<cap) active|=(__mmask8)(1u<<l);
        if(!active) break;

        alignas(64) long long ov[LANES],dv[LANES],sv[LANES],hv[LANES],iv[LANES],mmv[LANES];
        int tgt[LANES];
        for(int l=0;l<LANES;++l){
            int p=(active&(1u<<l))?pc[l]:0; const BInsn& ib=prog[(size_t)l*count+p];
            ov[l]=ib.op; dv[l]=ib.dst; sv[l]=ib.src; hv[l]=ib.shift; iv[l]=(long long)ib.imm;
            mmv[l]=(long long)(uint64_t)ib.memMask; tgt[l]=ib.target;
        }
        __m512i op=_mm512_loadu_si512(ov), dst=_mm512_loadu_si512(dv), src=_mm512_loadu_si512(sv),
                sh=_mm512_loadu_si512(hv), im=_mm512_loadu_si512(iv), memMask=_mm512_loadu_si512(mmv);

        __m512i s=selectReg(r,src), d=selectReg(r,dst), nd=d;
        auto OP=[&](BOp o){ return _mm512_cmpeq_epi64_mask(op,_mm512_set1_epi64((long long)o)); };
        nd=_mm512_mask_mov_epi64(nd, OP(B_IADD_RS), _mm512_add_epi64(d,_mm512_add_epi64(_mm512_sllv_epi64(s,sh),im)));
        nd=_mm512_mask_mov_epi64(nd, OP(B_ISUB_R),  _mm512_sub_epi64(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IMUL_R),  _mm512_mullo_epi64(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_INEG_R),  _mm512_add_epi64(_mm512_xor_si512(d,_mm512_set1_epi64(-1)),_mm512_set1_epi64(1)));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IXOR_R),  _mm512_xor_si512(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IROR_R),  _mm512_rorv_epi64(d,_mm512_and_si512(s,_mm512_set1_epi64(63))));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IROL_R),  _mm512_rolv_epi64(d,_mm512_and_si512(s,_mm512_set1_epi64(63))));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IMULH_R), mulhi_epu64(d,s));
        nd=_mm512_mask_mov_epi64(nd, OP(B_ISMULH_R),mulhi_epi64(d,s));

        __mmask8 cbm=OP(B_CBRANCH);
        __m512i d2=_mm512_add_epi64(d,im);         // CBRANCH: dst += imm
        nd=_mm512_mask_mov_epi64(nd, cbm, d2);

        __mmask8 swm=OP(B_ISWAP_R) & _mm512_cmpneq_epi64_mask(dst,src);
        nd=_mm512_mask_mov_epi64(nd, swm, s);

        scatterReg(r, dst, nd, active);
        scatterReg(r, src, d,  swm & active);

        __mmask8 taken = cbm & active &
            _mm512_cmpeq_epi64_mask(_mm512_and_si512(d2, memMask), _mm512_setzero_si512());
        for(int l=0;l<LANES;++l){
            if(!(active&(1u<<l))) continue;
            ++steps[l];
            pc[l] = (taken&(1u<<l)) ? tgt[l] : (pc[l]+1);
        }
    }

    for (int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)regs[k], r[k]);
}

// increment 3: per-lane memory ops (loads + ISTORE), each lane its own scratchpad.
void runPerLaneMem(uint64_t regs[IREGS][LANES], const BInsn* prog, int count, uint64_t* sp, uint64_t spWords)
{
    __m512i r[IREGS];
    for (int k=0;k<IREGS;++k) r[k]=_mm512_loadu_si512((const void*)regs[k]);
    const __m512i laneBase = _mm512_set_epi64(
        (long long)(7*spWords),(long long)(6*spWords),(long long)(5*spWords),(long long)(4*spWords),
        (long long)(3*spWords),(long long)(2*spWords),(long long)(1*spWords),0LL);

    for (int pos=0; pos<count; ++pos) {
        alignas(64) long long ov[LANES],dv[LANES],sv[LANES],iv[LANES],mmv[LANES];
        for (int l=0;l<LANES;++l){ const BInsn& ib=prog[(size_t)l*count+pos];
            ov[l]=ib.op; dv[l]=ib.dst; sv[l]=ib.src; iv[l]=(long long)ib.imm; mmv[l]=(long long)(uint64_t)ib.memMask; }
        __m512i op=_mm512_loadu_si512(ov), dst=_mm512_loadu_si512(dv), src=_mm512_loadu_si512(sv),
                imm=_mm512_loadu_si512(iv), mask=_mm512_loadu_si512(mmv);
        auto OP=[&](BOp o){ return _mm512_cmpeq_epi64_mask(op,_mm512_set1_epi64((long long)o)); };

        __m512i s=selectReg(r,src), d=selectReg(r,dst);
        __mmask8 stm=OP(B_ISTORE), ldm=(__mmask8)(~stm);

        __m512i addrS=_mm512_and_si512(_mm512_add_epi64(d,imm),mask);
        __m512i widxS=_mm512_add_epi64(_mm512_srli_epi64(addrS,3),laneBase);
        _mm512_mask_i64scatter_epi64((long long*)sp, stm, widxS, s, 8);

        __m512i addrL=_mm512_and_si512(_mm512_add_epi64(s,imm),mask);
        __m512i widxL=_mm512_add_epi64(_mm512_srli_epi64(addrL,3),laneBase);
        __m512i v=_mm512_mask_i64gather_epi64(_mm512_setzero_si512(), ldm, widxL, (const long long*)sp, 8);

        __m512i nd=d;
        nd=_mm512_mask_mov_epi64(nd, OP(B_IADD_M),  _mm512_add_epi64(d,v));
        nd=_mm512_mask_mov_epi64(nd, OP(B_ISUB_M),  _mm512_sub_epi64(d,v));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IMUL_M),  _mm512_mullo_epi64(d,v));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IMULH_M), mulhi_epu64(d,v));
        nd=_mm512_mask_mov_epi64(nd, OP(B_ISMULH_M),mulhi_epi64(d,v));
        nd=_mm512_mask_mov_epi64(nd, OP(B_IXOR_M),  _mm512_xor_si512(d,v));
        scatterReg(r, dst, nd, ldm);
    }

    for (int k=0;k<IREGS;++k) _mm512_storeu_si512((void*)regs[k], r[k]);
}

// increment 4: per-lane float register ops. selectRegF/scatterRegF are the float
// analog of selectReg — gather/scatter each lane's F[idx[lane]] (lo+hi) across the
// FREGS float-register vectors.
static inline BFReg selectRegF(const BFReg reg[FREGS], __m512i idx)
{
    BFReg sel = reg[0];
    for (int k=1;k<FREGS;++k){ __mmask8 mk=_mm512_cmpeq_epi64_mask(idx,_mm512_set1_epi64(k));
        sel.lo=_mm512_mask_mov_pd(sel.lo,mk,reg[k].lo); sel.hi=_mm512_mask_mov_pd(sel.hi,mk,reg[k].hi); }
    return sel;
}
static inline void scatterRegF(BFReg reg[FREGS], __m512i idx, BFReg val, __mmask8 active)
{
    for (int k=0;k<FREGS;++k){ __mmask8 mk=active & _mm512_cmpeq_epi64_mask(idx,_mm512_set1_epi64(k));
        reg[k].lo=_mm512_mask_mov_pd(reg[k].lo,mk,val.lo); reg[k].hi=_mm512_mask_mov_pd(reg[k].hi,mk,val.hi); }
}

void runPerLaneFloat(BFReg reg[FREGS], const FInsn* prog, int count)
{
    for (int pos=0; pos<count; ++pos) {
        alignas(64) long long ov[LANES],dv[LANES],sv[LANES];
        for (int l=0;l<LANES;++l){ const FInsn& in=prog[(size_t)l*count+pos]; ov[l]=in.op; dv[l]=in.dst; sv[l]=in.src; }
        __m512i op=_mm512_loadu_si512(ov), dst=_mm512_loadu_si512(dv), src=_mm512_loadu_si512(sv);
        auto OP=[&](FOp o){ return _mm512_cmpeq_epi64_mask(op,_mm512_set1_epi64((long long)o)); };

        BFReg s=selectRegF(reg,src), d=selectRegF(reg,dst), nd=d;
        { __mmask8 m=OP(F_FADD_R); nd.lo=_mm512_mask_add_pd(nd.lo,m,d.lo,s.lo); nd.hi=_mm512_mask_add_pd(nd.hi,m,d.hi,s.hi); }
        { __mmask8 m=OP(F_FSUB_R); nd.lo=_mm512_mask_sub_pd(nd.lo,m,d.lo,s.lo); nd.hi=_mm512_mask_sub_pd(nd.hi,m,d.hi,s.hi); }
        { __mmask8 m=OP(F_FMUL_R); nd.lo=_mm512_mask_mul_pd(nd.lo,m,d.lo,s.lo); nd.hi=_mm512_mask_mul_pd(nd.hi,m,d.hi,s.hi); }
        { __mmask8 m=OP(F_FSQRT_R); nd.lo=_mm512_mask_sqrt_pd(nd.lo,m,emask(d.lo)); nd.hi=_mm512_mask_sqrt_pd(nd.hi,m,emask(d.hi)); }
        { __mmask8 m=OP(F_FSCAL_R); nd.lo=_mm512_mask_mov_pd(nd.lo,m,fscal(d.lo)); nd.hi=_mm512_mask_mov_pd(nd.hi,m,fscal(d.hi)); }
        { __mmask8 m=OP(F_FSWAP_R); nd.lo=_mm512_mask_mov_pd(nd.lo,m,d.hi); nd.hi=_mm512_mask_mov_pd(nd.hi,m,d.lo); }
        scatterRegF(reg, dst, nd, (__mmask8)0xFF);
    }
}


// increment 5a: unified per-lane interpreter — 8 DIFFERENT programs, one per lane,
// per-lane pc, dispatching int/branch/memory/float/CFROUND together.
static inline BFReg selectRegF8(const BFReg reg[8], __m512i idx)
{
    BFReg sel = reg[0];
    for (int k=1;k<8;++k){ __mmask8 mk=_mm512_cmpeq_epi64_mask(idx,_mm512_set1_epi64(k));
        sel.lo=_mm512_mask_mov_pd(sel.lo,mk,reg[k].lo); sel.hi=_mm512_mask_mov_pd(sel.hi,mk,reg[k].hi); }
    return sel;
}
static inline void scatterRegF8(BFReg reg[8], __m512i idx, BFReg val, __mmask8 active)
{
    for (int k=0;k<8;++k){ __mmask8 mk=active & _mm512_cmpeq_epi64_mask(idx,_mm512_set1_epi64(k));
        reg[k].lo=_mm512_mask_mov_pd(reg[k].lo,mk,val.lo); reg[k].hi=_mm512_mask_mov_pd(reg[k].hi,mk,val.hi); }
}

void runBytecodeVecPerLane(__m512i r[IREGS], BFReg F[8], const BFReg A[4],
                           uint64_t* sp, uint64_t spWords,
                           __m512d eMaskLo, __m512d eMaskHi,
                           const FullProg* prog, int rmode[LANES])
{
    const __m512i laneBase = _mm512_set_epi64(
        (long long)(7*spWords),(long long)(6*spWords),(long long)(5*spWords),(long long)(4*spWords),
        (long long)(3*spWords),(long long)(2*spWords),(long long)(1*spWords),0LL);
    const __m512d mantMask = _mm512_castsi512_pd(_mm512_set1_epi64((long long)E_MANTISSA_MASK));
    const int count = prog[0].count;
    const bool tweak = RandomX_CurrentConfig.Tweak_V2_CFROUND;

    int pc[LANES]; for(int l=0;l<LANES;++l) pc[l]=0;
    long steps[LANES]; for(int l=0;l<LANES;++l) steps[l]=0;
    const long cap=(long)count*64;

    for(;;){
        __mmask8 active=0;
        for(int l=0;l<LANES;++l) if(pc[l]>=0 && pc[l]<count && steps[l]<cap) active|=(__mmask8)(1u<<l);
        if(!active) break;

        alignas(64) long long kv[LANES],ov[LANES],dv[LANES],sv[LANES],hv[LANES],iv[LANES],mmv[LANES],
                              siv[LANES],svv[LANES],fpv[LANES],fdv[LANES],fsv[LANES],mav[LANES],fiv[LANES],fmv[LANES];
        int tgt[LANES]; uint64_t rot[LANES];
        for(int l=0;l<LANES;++l){
            int p=(active&(1u<<l))?pc[l]:0; const FullInsn& fi=prog[l].ins[p]; const BInsn& ib=fi.ib;
            kv[l]=fi.kind; ov[l]=ib.op; dv[l]=ib.dst; sv[l]=ib.src; hv[l]=ib.shift; iv[l]=(long long)ib.imm;
            mmv[l]=(long long)(uint64_t)ib.memMask; tgt[l]=ib.target; siv[l]=ib.srcImm?1:0; svv[l]=(long long)ib.srcVal;
            fpv[l]=fi.fop; fdv[l]=fi.fdst; fsv[l]=fi.fsrc; mav[l]=fi.maddr; fiv[l]=(long long)fi.fimm; fmv[l]=(long long)(uint64_t)fi.fmask;
            rot[l]=(unsigned)fi.fimm;
        }
        __m512i kind=_mm512_loadu_si512(kv), op=_mm512_loadu_si512(ov), dst=_mm512_loadu_si512(dv),
                src=_mm512_loadu_si512(sv), sh=_mm512_loadu_si512(hv), im=_mm512_loadu_si512(iv),
                memMask=_mm512_loadu_si512(mmv), srcImm=_mm512_loadu_si512(siv), srcVal=_mm512_loadu_si512(svv),
                fop=_mm512_loadu_si512(fpv), fdst=_mm512_loadu_si512(fdv), fsrc=_mm512_loadu_si512(fsv),
                maddr=_mm512_loadu_si512(mav), fmask=_mm512_loadu_si512(fmv), fimm=_mm512_loadu_si512(fiv);

        __mmask8 kInt   = active & _mm512_cmpeq_epi64_mask(kind,_mm512_set1_epi64(0));
        __mmask8 kFloat = active & _mm512_cmpeq_epi64_mask(kind,_mm512_set1_epi64(1));
        __mmask8 kCf    = active & _mm512_cmpeq_epi64_mask(kind,_mm512_set1_epi64(3));

        // ---- CFROUND ----
        if(kCf){
            alignas(64) uint64_t rv[LANES]; _mm512_storeu_si512(rv, selectReg(r, maddr));
            for(int l=0;l<LANES;++l) if(kCf&(1u<<l)){
                uint64_t v=s_rotr(rv[l], (unsigned)rot[l]);
                if(!tweak || (v&60)==0) rmode[l]=(int)(v&3);
            }
        }

        // ---- integer / branch / memory (kind==0) ----
        auto IOP=[&](BOp o){ return kInt & _mm512_cmpeq_epi64_mask(op,_mm512_set1_epi64((long long)o)); };
        __m512i sReg=selectReg(r,src);
        __m512i s=_mm512_mask_mov_epi64(sReg, _mm512_cmpneq_epi64_mask(srcImm,_mm512_setzero_si512()), srcVal);
        __m512i d=selectReg(r,dst);
        __m512i nd=d;
        nd=_mm512_mask_mov_epi64(nd, IOP(B_IADD_RS), _mm512_add_epi64(d,_mm512_add_epi64(_mm512_sllv_epi64(s,sh),im)));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_ISUB_R),  _mm512_sub_epi64(d,s));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_IMUL_R),  _mm512_mullo_epi64(d,s));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_INEG_R),  _mm512_add_epi64(_mm512_xor_si512(d,_mm512_set1_epi64(-1)),_mm512_set1_epi64(1)));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_IXOR_R),  _mm512_xor_si512(d,s));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_IROR_R),  _mm512_rorv_epi64(d,_mm512_and_si512(s,_mm512_set1_epi64(63))));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_IROL_R),  _mm512_rolv_epi64(d,_mm512_and_si512(s,_mm512_set1_epi64(63))));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_IMULH_R), mulhi_epu64(d,s));
        nd=_mm512_mask_mov_epi64(nd, IOP(B_ISMULH_R),mulhi_epi64(d,s));
        // memory loads
        __mmask8 ldm = IOP(B_IADD_M)|IOP(B_ISUB_M)|IOP(B_IMUL_M)|IOP(B_IMULH_M)|IOP(B_ISMULH_M)|IOP(B_IXOR_M);
        {
            __m512i addr=_mm512_and_si512(_mm512_add_epi64(s,im),memMask);
            __m512i widx=_mm512_add_epi64(_mm512_srli_epi64(addr,3),laneBase);
            __m512i v=_mm512_mask_i64gather_epi64(_mm512_setzero_si512(), ldm, widx, (const long long*)sp, 8);
            nd=_mm512_mask_mov_epi64(nd, IOP(B_IADD_M),  _mm512_add_epi64(d,v));
            nd=_mm512_mask_mov_epi64(nd, IOP(B_ISUB_M),  _mm512_sub_epi64(d,v));
            nd=_mm512_mask_mov_epi64(nd, IOP(B_IMUL_M),  _mm512_mullo_epi64(d,v));
            nd=_mm512_mask_mov_epi64(nd, IOP(B_IMULH_M), mulhi_epu64(d,v));
            nd=_mm512_mask_mov_epi64(nd, IOP(B_ISMULH_M),mulhi_epi64(d,v));
            nd=_mm512_mask_mov_epi64(nd, IOP(B_IXOR_M),  _mm512_xor_si512(d,v));
        }
        // CBRANCH: d2 = d + imm
        __mmask8 cbm = IOP(B_CBRANCH);
        __m512i d2=_mm512_add_epi64(d,im);
        nd=_mm512_mask_mov_epi64(nd, cbm, d2);
        // ISWAP
        __mmask8 swm = IOP(B_ISWAP_R) & _mm512_cmpneq_epi64_mask(dst,src);
        nd=_mm512_mask_mov_epi64(nd, swm, s);
        // ISTORE: addr=(d+imm)&memMask ; store s
        __mmask8 stm = IOP(B_ISTORE);
        {
            __m512i addr=_mm512_and_si512(_mm512_add_epi64(d,im),memMask);
            __m512i widx=_mm512_add_epi64(_mm512_srli_epi64(addr,3),laneBase);
            _mm512_mask_i64scatter_epi64((long long*)sp, stm, widx, s, 8);
        }
        // write back integer results (all kInt lanes except pure-store; store lanes' nd==d so harmless)
        scatterReg(r, dst, nd, kInt);
        scatterReg(r, src, d,  swm);

        // ---- float (kind==1) ----
        if(kFloat){
            auto FOP=[&](FBOp o){ return kFloat & _mm512_cmpeq_epi64_mask(fop,_mm512_set1_epi64((long long)o)); };
            BFReg fd=selectRegF8(F,fdst);
            BFReg fa=selectRegF(A,fsrc);
            // memory operand for FADD_M/FSUB_M/FDIV_M
            __m512i faddr=_mm512_and_si512(_mm512_add_epi64(selectReg(r,maddr),fimm),fmask);
            __m512i fwidx=_mm512_add_epi64(_mm512_srli_epi64(faddr,3),laneBase);
            __mmask8 mMem = FOP(FB_FADD_M)|FOP(FB_FSUB_M)|FOP(FB_FDIV_M);
            __m512i mw=_mm512_mask_i64gather_epi64(_mm512_setzero_si512(), mMem, fwidx, (const long long*)sp, 8);
            __m512d mlo,mhi; cvtPackedIntPD(mw, mlo, mhi);
            __m512d dlo=_mm512_or_pd(_mm512_and_pd(mlo,mantMask),eMaskLo);   // FDIV divisor mask
            __m512d dhi=_mm512_or_pd(_mm512_and_pd(mhi,mantMask),eMaskHi);
            // operand select: R ops use a-reg fa; ADD_M/SUB_M use raw mlo/mhi; DIV_M uses masked dlo/dhi
            __mmask8 addm=FOP(FB_FADD_M), subm=FOP(FB_FSUB_M), divm=FOP(FB_FDIV_M);
            __m512d slo=fa.lo, shi=fa.hi;
            slo=_mm512_mask_mov_pd(slo, addm|subm, mlo); shi=_mm512_mask_mov_pd(shi, addm|subm, mhi);
            slo=_mm512_mask_mov_pd(slo, divm, dlo);      shi=_mm512_mask_mov_pd(shi, divm, dhi);

            BFReg nf=fd;
            // rounding-independent
            __mmask8 swf=FOP(FB_FSWAP);
            nf.lo=_mm512_mask_mov_pd(nf.lo, swf, fd.hi); nf.hi=_mm512_mask_mov_pd(nf.hi, swf, fd.lo);
            __mmask8 scf=FOP(FB_FSCAL_R);
            nf.lo=_mm512_mask_mov_pd(nf.lo, scf, fscal(fd.lo)); nf.hi=_mm512_mask_mov_pd(nf.hi, scf, fscal(fd.hi));
            // rounding-sensitive: group by per-lane mode
            __mmask8 addr_ = FOP(FB_FADD_R)|FOP(FB_FADD_M);
            __mmask8 subr_ = FOP(FB_FSUB_R)|FOP(FB_FSUB_M);
            __mmask8 mulr_ = FOP(FB_FMUL_R);
            __mmask8 divr_ = FOP(FB_FDIV_M);
            __mmask8 sqrr_ = FOP(FB_FSQRT_R);
            for(int md=0;md<4;++md){
                alignas(64) long long rmv[LANES]; for(int l=0;l<LANES;++l) rmv[l]=rmode[l];
                __mmask8 mm=_mm512_cmpeq_epi64_mask(_mm512_loadu_si512(rmv),_mm512_set1_epi64(md));
                __mmask8 a=addr_&mm, b=subr_&mm, c=mulr_&mm, e=divr_&mm, g=sqrr_&mm;
                if(!(a|b|c|e|g)) continue;
                _mm_setcsr(0x9FC0u|((unsigned)md<<13));
                if(a){ nf.lo=_mm512_mask_add_pd(nf.lo,a,fd.lo,slo); nf.hi=_mm512_mask_add_pd(nf.hi,a,fd.hi,shi); }
                if(b){ nf.lo=_mm512_mask_sub_pd(nf.lo,b,fd.lo,slo); nf.hi=_mm512_mask_sub_pd(nf.hi,b,fd.hi,shi); }
                if(c){ nf.lo=_mm512_mask_mul_pd(nf.lo,c,fd.lo,slo); nf.hi=_mm512_mask_mul_pd(nf.hi,c,fd.hi,shi); }
                if(e){ nf.lo=_mm512_mask_div_pd(nf.lo,e,fd.lo,slo); nf.hi=_mm512_mask_div_pd(nf.hi,e,fd.hi,shi); }
                if(g){ nf.lo=_mm512_mask_sqrt_pd(nf.lo,g,fd.lo);    nf.hi=_mm512_mask_sqrt_pd(nf.hi,g,fd.hi); }
            }
            scatterRegF8(F, fdst, nf, kFloat);
        }

        // ---- per-lane pc update ----
        __mmask8 taken = cbm & _mm512_cmpeq_epi64_mask(_mm512_and_si512(d2,memMask),_mm512_setzero_si512());
        for(int l=0;l<LANES;++l){
            if(!(active&(1u<<l))) continue;
            ++steps[l];
            pc[l] = (taken&(1u<<l)) ? (tgt[l]+1) : (pc[l]+1);
        }
    }
}

#if defined(__GNUC__)
#   pragma GCC pop_options
#endif

} // namespace batchedx
} // namespace xmrig
