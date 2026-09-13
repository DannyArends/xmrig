if (WITH_BATCHEDX)
    add_definitions(/DXMRIG_FEATURE_BATCHEDX)

    list(APPEND HEADERS_CRYPTO
        src/crypto/batchedx/BatchedVm.h
    )
    list(APPEND SOURCES_CRYPTO
        src/crypto/batchedx/BatchedVm.cpp
    )
endif()