#pragma once

#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>

namespace NCB::NModelEvaluation::AVX {
    
    enum class EEvaluationMethod { NON_SSE, SSE3_BS128, AVX2_BS256, AVX2_BS128, AVX2_BS64, 
                                   AVX512_BS512, AVX512_BS256, AVX512_BS128, AVX512_BS64,
                                   AVX512_BS512_SHUFFLE, AVX512_BS256_SHUFFLE, AVX512_BS128_SHUFFLE,
                                   AVX2_BS256_GATHER, AVX2_BS128_GATHER,
                                   AVX512_BS512_GATHER, AVX512_BS256_GATHER, AVX512_BS128_GATHER,
                                   NON_SSE_FP16, AVX512_BS512_FP16, AVX512_BS256_FP16, AVX512_BS128_FP16,
                                   AVX512_BS512_SHUFFLE_FP16, AVX512_BS256_SHUFFLE_FP16, AVX512_BS128_SHUFFLE_FP16 };

    TModelEvaluatorPtr CreateAvxEvaluator(EEvaluationMethod method, const TFullModel& fullModel);

}
