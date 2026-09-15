# batchedX

8-wide AVX-512 RandomX. Runs 8 nonces at once, one per SIMD lane, each with its own
program and scratchpad. One `__m512i` per integer register, one `BFReg` per float
register, all 8 lanes in lockstep.

It's bit-exact against RandomX in xmrig: full instruction set, CFROUND, memory, the 
whole hash. It was tested and mines real shares on P2Pool. It works, it's slow.

## What's where

- `BatchedVm.h` : public API, `batchedHash8()` and the `verify*()` calls
- `BatchedOps.h` : the AVX-512 primitives and the `BFReg` float type
- `BatchedInternal.h` : shared types, helpers, cross-file decls
- `batched_translate.cpp` : RandomX bytecode into my `FullInsn`/`BInsn` form
- `batched_exec.cpp` : the actual SIMD engine
- `batched_ref.cpp` : scalar oracle, verify only
- `batched_verify.cpp` : the verify stages plus `batchedHash8()`

The four main functions:

- `batchedHash8(dataset, scratchpad, spWords, blobs[8], inputSize, out)` : hand it 8
  blobs, get 8 hashes. The hasher.
- `batchedMine(...)` : the mining hook. Decides whether the batched path applies (N==8,
  fast mode, plain rx/0) and calls `batchedHash8`. All the mining logic lives here so the
  seam into xmrig stays tiny.
- `runBytecodeVecPerLane(...)` : the per-lane interpreter. 8 different programs, each
  on its own pc. Int, branch, memory, float, CFROUND.
- `runBatchedExecutePerLane(...)` : the VM loop around it.

Per-lane register access is `selectReg`/`scatterReg` (and the float versions).

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

## Mine

Mining goes through xmrig's CPU worker. Batched RandomX runs at intensity 8, and calls 
`batchedMine`). Build with `-DWITH_BATCHEDX=ON`, then drop a `config.json` next to the 
compiled binary:

```json
{
    "autosave": false,
    "cpu": {
        "enabled": true,
        "huge-pages": true,
        "rx": { "intensity": 8, "threads": 8, "affinity": -1 }
    },
    "randomx": { "mode": "fast", "1gb-pages": false },
    "pools": [
        { "url": "127.0.0.1:3333", "user": "x", "pass": "x", "algo": "rx/0" }
    ]
}
```

```
./xmrig -c config.json
```

Two settings: `intensity: 8` picks the batched path (`CpuWorker<8>`), and `mode: fast` 
gives it the full dataset it needs. Only `rx/0`, miner-signature and commitment variants 
fall back to the JIT. Wait for `accepted`, then you're mining 8-wide. Set `threads` to 
your core count. And yeah, it's slow.

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
