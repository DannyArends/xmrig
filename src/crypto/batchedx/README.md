# batchedX

8-wide AVX-512 RandomX. Runs 8 nonces at once, one per SIMD lane, each with its own
program and scratchpad. One `__m512i` per integer register, one `BFReg` per float
register, all 8 lanes in lockstep.

It's bit-exact against randomX in xmrig: full instruction set, CFROUND, memory, the 
whole hash. It was tested and mines real shares on P2Pool. It works. It's also slow.

## What's where

- `BatchedVm.h` : public API, `batchedHash8()` and the `verify*()` calls
- `BatchedOps.h` : the AVX-512 primitives and the `BFReg` float type
- `BatchedInternal.h` : shared types, helpers, cross-file decls
- `batched_translate.cpp` : RandomX bytecode into my `FullInsn`/`BInsn` form
- `batched_exec.cpp` : the actual SIMD engine
- `batched_ref.cpp` : scalar oracle, verify only
- `batched_verify.cpp` : the verify stages plus `batchedHash8()`

The three things that matter:

- `batchedHash8(dataset, scratchpad, spWords, blobs[8], inputSize, out)` : hand it 8
  blobs, get 8 hashes. That's the whole miner-facing surface.
- `runBytecodeVecPerLane(...)` : the per-lane interpreter. 8 different programs, each
  on its own pc. Int, branch, memory, float, CFROUND.
- `runBatchedExecutePerLane(...)` : the VM loop around it.

Per-lane register access is `selectReg`/`scatterReg` (and the float versions). That's
where a chunk of the cost lives, see below.

## Build and check

```
cmake -DWITH_BATCHEDX=ON ..
```

Needs AVX-512F + DQ. To prove it's correct:

```
./xmrig --batchedx-verify
```

That runs Stages 1 to 14, all bit-exact vs xmrig. 1 to 8 are the shared-program path,
9 to 14 are the real thing: 8 different nonces through the full per-lane hash vs
`randomx_calculate_hash`.

## On the speed

Don't expect it to beat the JIT. It's roughly an order of magnitude slower, and that's
RandomX doing its job, not a bug. Every nonce gets a different program on purpose, so 
batching pays two costs the JIT never sees:

1. Register selection. Each lane wants a different register, so every instruction burns
   extra vector ops shuffling operands in and out. The JIT bakes the indices into
   native code for free.
2. Opcode divergence. The 8 lanes rarely agree on the opcode, so one instruction turns
   into several masked passes.

Those are exactly the limits build into RandomX to try and defeat batching of nonces. 
No compiler flag makes them go away. 

## Future direction

If I ever come back to this, the one lever left is a struct-of-arrays program layout to 
kill the per-instruction marshalling.

Built because nobody had done 8-wide AVX-512 RandomX. Now somebody has.
