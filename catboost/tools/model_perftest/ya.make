

PROGRAM()

PEERDIR(
    catboost/tools/model_perftest/lib
    library/cpp/getopt
)

CFLAGS(
    -mavx512f
    -mavx512bw
    -mavx512cd
    -mavx512dq
    -mavx512vl
)

END()
