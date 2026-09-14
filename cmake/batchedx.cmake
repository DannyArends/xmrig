if (WITH_BATCHEDX)
    add_definitions(/DXMRIG_FEATURE_BATCHEDX)

    list(APPEND HEADERS_CRYPTO
        src/crypto/batchedx/BatchedVm.h
        src/crypto/batchedx/BatchedOps.h
        src/crypto/batchedx/BatchedInternal.h
    )
    list(APPEND SOURCES_CRYPTO
        src/crypto/batchedx/batched_translate.cpp
        src/crypto/batchedx/batched_exec.cpp
        src/crypto/batchedx/batched_ref.cpp
        src/crypto/batchedx/batched_verify.cpp
    )
endif()
