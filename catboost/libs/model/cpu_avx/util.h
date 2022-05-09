#pragma once

#include "evaluator_interface.h"

#if defined _ubsan_enabled_
#define NFORCE_UNROLL
#else
#define NFORCE_UNROLL _Pragma("unroll")
#endif

namespace NCB::NModelEvaluation::AVX {

    template<EEvaluationMethod EvaluationMethod>
    struct IConsts{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 16;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX2_BS256>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 256;
        constexpr static size_t SUBBLOCK_SIZE = 32;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX2_BS128>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 32;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX2_BS64>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 64;
        constexpr static size_t SUBBLOCK_SIZE = 32;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS512>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 512;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS256>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 256;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS128>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS64>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 64;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS512_SHUFFLE>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 512;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS256_SHUFFLE>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 256;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS128_SHUFFLE>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX2_BS256_GATHER>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 256;
        constexpr static size_t SUBBLOCK_SIZE = 32;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX2_BS128_GATHER>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 32;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS512_GATHER>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 512;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS256_GATHER>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 256;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS128_GATHER>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS512_FP16>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 512;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS256_FP16>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 256;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS128_FP16>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 512;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 256;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };

    template<>
    struct IConsts<EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16>{
        constexpr static size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
        constexpr static size_t SUBBLOCK_SIZE = 64;
    };


    template <class X>
    inline X* GetAligned(X* val) {
        uintptr_t off = ((uintptr_t)val) % 64;
        val = (X*)((ui8*)val + (64 - off) % 64);
        return val;
    }

    template <auto Start, auto End, auto Inc, class F>
    constexpr void ConstexprFor(F&& f)
    {
        if constexpr (Start < End)
        {
            f(std::integral_constant<decltype(Start), Start>());
            ConstexprFor<Start + Inc, End, Inc>(f);
        }
    }

}
