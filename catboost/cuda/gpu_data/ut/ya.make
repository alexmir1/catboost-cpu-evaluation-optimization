

IF (AUTOCHECK)
    LIBRARY()
ELSE()
    UNITTEST()
    SIZE(MEDIUM)
    IF (OS_LINUX AND NOT ARCH_AARCH64)
        ALLOCATOR(TCMALLOC_256K)
    ELSE()
        ALLOCATOR(J)
    ENDIF()
ENDIF()

SRCS(
    test_bin_builder.cpp
    test_binarization.cpp
    test_uniform_binarization.cpp
)

PEERDIR(
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/ut_helpers
    catboost/libs/helpers
    catboost/private/libs/quantization
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
