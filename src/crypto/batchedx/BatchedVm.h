/* XMRig — batched AVX-512 RandomX (8-lane), header.
 *
 * Compiled in only when WITH_BATCHEDX=ON (XMRIG_FEATURE_BATCHEDX).
 *
 * Stage 1: an 8-lane interpreter for the RandomX INTEGER register ops. Registers are
 * held as one __m512i per register index (lane = nonce). Each op is applied across all
 * 8 lanes at once. Verified bit-exact against scalar semantics by verifyIntegerOps().
 */

#ifndef XMRIG_BATCHEDVM_H
#define XMRIG_BATCHEDVM_H


#include <cstddef>
#include <cstdint>


namespace xmrig {


namespace batchedx {

static constexpr int LANES = 8;      // AVX-512 = 8 x 64-bit lanes
static constexpr int IREGS = 8;      // RandomX integer registers per lane

// A batched instruction (integer register ops only, for Stage 1). Register operands
// are indices 0..7 (not pointers), so the same decoded instruction drives all lanes.
enum BOp : uint8_t {
    B_IADD_RS, B_ISUB_R, B_IMUL_R, B_INEG_R, B_IXOR_R, B_IROR_R, B_IROL_R, B_ISWAP_R,
    B_IMULH_R, B_ISMULH_R,
    B_IADD_M, B_ISUB_M, B_IMUL_M, B_IMULH_M, B_ISMULH_M, B_IXOR_M, B_ISTORE,
    B_CBRANCH
};

struct BInsn {
    BOp      op;
    uint8_t  dst;     // 0..7
    uint8_t  src;     // 0..7
    uint8_t  shift;   // IADD_RS shift (0..3)
    uint64_t imm;     // IADD_RS / memory immediate
    uint32_t memMask; // scratchpad address mask (memory ops) / condition mask (CBRANCH)
    int16_t  target;  // CBRANCH backward-jump target pc
    bool     srcImm;  // true = source operand is the immediate srcVal (src==dst form)
    uint64_t srcVal;  // immediate source value when srcImm
};

/* Run a batched integer-only program. regs[k] holds register k across 8 lanes
 * (regs is IREGS x LANES uint64). Applies each instruction across all lanes. */
void runIntegerProgram(uint64_t regs[IREGS][LANES], const BInsn* prog, int count);

/* Self-test: random integer-only programs, batched vs scalar reference, bit-exact.
 * Returns 0 on all-pass. */
int verifyIntegerOps();

/* Stage 3: float register ops self-test. Returns 0 on all-pass. */
int verifyFloatOps();

/* Stage 4: memory ops (scratchpad gather/scatter) self-test. Returns 0 on all-pass. */
int verifyMemoryOps();

/* Stage 5: CBRANCH (per-lane PC + masked execution) self-test. Returns 0 on all-pass. */
int verifyBranchOps();

/* Stage 6a: real-program (int/branch) verify. Returns 0 on all-pass. */
int verifyProgram();

/* Stage 6b: full real-program verify vs executeBytecode. Returns 0 on all-pass. */
int verifyProgramFull();
int verifyProgramFullRounded();
int verifyBatchedExecute();
int verifyBatchedHash();
int verifyPerLaneInt();
int verifyPerLaneBranch();

} // namespace batchedx


} // namespace xmrig


#endif // XMRIG_BATCHEDVM_H
