#include "perftest_module.h"
#include <catboost/libs/model/cpu_avx/evaluator_interface.h>

class TBaseCatboostModule : public TBasePerftestModule {
public:
    TBaseCatboostModule() = default;

    int GetComparisonPriority(EPerftestModuleDataLayout layout) const override {
        if (layout == EPerftestModuleDataLayout::ObjectsFirst) {
            return Priority + 1;
        }
        return Priority;
    }

    bool SupportsLayout(EPerftestModuleDataLayout ) const final {
        return true;
    }

    double Do(EPerftestModuleDataLayout layout, TConstArrayRef<TConstArrayRef<float>> features) final {
        if (layout == EPerftestModuleDataLayout::ObjectsFirst) {
            ResultsHolder.resize(features.size());
            Timer.Reset();
            ModelEvaluator->CalcFlat(features, ResultsHolder);
            Y_UNUSED(ResultsHolder);
            return Timer.Passed();
        } else {
            ResultsHolder.resize(features[0].size());
            Timer.Reset();
            ModelEvaluator->CalcFlatTransposed(features, ResultsHolder);
            Y_UNUSED(ResultsHolder);
            return Timer.Passed();
        }
    }

    TVector<double> GetApplyResult(EPerftestModuleDataLayout layout, TConstArrayRef<TConstArrayRef<float>> features) final {
        Do(layout, features);
        return ResultsHolder;
    }

    TString GetName(TMaybe<EPerftestModuleDataLayout> layout) const final {
        if (!layout.Defined()) {
            return BaseName;
        } else if (*layout == EPerftestModuleDataLayout::ObjectsFirst) {
            return BaseName + " objects order";
        } else {
            return BaseName + " features order";
        }
    }

protected:
    NCB::NModelEvaluation::TModelEvaluatorPtr ModelEvaluator;
    int Priority = 0;
    TString BaseName;
    TVector<double> ResultsHolder;
};

class TCPUCatboostModule : public TBaseCatboostModule {
public:
    TCPUCatboostModule(const TFullModel& model) {
        Priority = 10;
        ModelEvaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::CPU, model);
        Y_ENSURE(ModelEvaluator, "Failed to create catboost cpu elevaluator");
        BaseName = "catboost cpu";
    }
};

TPerftestModuleFactory::TRegistrator<TCPUCatboostModule> CPUCatboostModuleRegistar("CPUCatboost");


class TAVXModule_NON_SSE: public TBaseCatboostModule {
public:
    TAVXModule_NON_SSE(const TFullModel& model) {
        Priority = 0;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::NON_SSE, model);
        BaseName = "NON SSE";
    }
};

//TPerftestModuleFactory::TRegistrator<TAVXModule_NON_SSE> AVXModuleRegistar_NON_SSE("NON_SSE");


class TAVXModule_SSE3_BS128 : public TBaseCatboostModule {
public:
    TAVXModule_SSE3_BS128(const TFullModel& model) {
        Priority = -10;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::SSE3_BS128, model);
        BaseName = "SSE3 Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_SSE3_BS128> AVXModuleRegistar_SSE3_BS128("SSE3_BS128");


class TAVXModule_AVX2_BS256 : public TBaseCatboostModule {
public:
    TAVXModule_AVX2_BS256(const TFullModel& model) {
        Priority = -20;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX2_BS256, model);
        BaseName = "AVX2 Block Size 256";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX2_BS256> AVXModuleRegistar_AVX2_BS256("AVX2_BS256");


class TAVXModule_AVX2_BS128 : public TBaseCatboostModule {
public:
    TAVXModule_AVX2_BS128(const TFullModel& model) {
        Priority = -22;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX2_BS128, model);
        BaseName = "AVX2 Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX2_BS128> AVXModuleRegistar_AVX2_BS128("AVX2_BS128");


class TAVXModule_AVX2_BS64 : public TBaseCatboostModule {
public:
    TAVXModule_AVX2_BS64(const TFullModel& model) {
        Priority = -24;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX2_BS64, model);
        BaseName = "AVX2 Block Size 64";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX2_BS64> AVXModuleRegistar_AVX2_BS64("AVX2_BS64");


class TAVXModule_AVX512_BS512 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS512(const TFullModel& model) {
        Priority = -30;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS512, model);
        BaseName = "AVX512 Block Size 512";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS512> AVXModuleRegistar_AVX512_BS512("AVX512_BS512");


class TAVXModule_AVX512_BS256 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS256(const TFullModel& model) {
        Priority = -32;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS256, model);
        BaseName = "AVX512 Block Size 256";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS256> AVXModuleRegistar_AVX512_BS256("AVX512_BS256");


class TAVXModule_AVX512_BS128 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS128(const TFullModel& model) {
        Priority = -34;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS128, model);
        BaseName = "AVX512 Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS128> AVXModuleRegistar_AVX512_BS128("AVX512_BS128");


class TAVXModule_AVX512_BS64 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS64(const TFullModel& model) {
        Priority = -36;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS64, model);
        BaseName = "AVX512 Block Size 64";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS64> AVXModuleRegistar_AVX512_BS64("AVX512_BS64");


class TAVXModule_AVX512_BS512_SHUFFLE : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS512_SHUFFLE(const TFullModel& model) {
        Priority = -40;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS512_SHUFFLE, model);
        BaseName = "AVX512 SHUFFLE Block Size 512";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS512_SHUFFLE> AVXModuleRegistar_AVX512_BS512_SHUFFLE("AVX512_BS512_SHUFFLE");


class TAVXModule_AVX512_BS256_SHUFFLE : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS256_SHUFFLE(const TFullModel& model) {
        Priority = -42;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS256_SHUFFLE, model);
        BaseName = "AVX512 SHUFFLE Block Size 256";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS256_SHUFFLE> AVXModuleRegistar_AVX512_BS256_SHUFFLE("AVX512_BS256_SHUFFLE");


class TAVXModule_AVX512_BS128_SHUFFLE : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS128_SHUFFLE(const TFullModel& model) {
        Priority = -44;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS128_SHUFFLE, model);
        BaseName = "AVX512 SHUFFLE Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS128_SHUFFLE> AVXModuleRegistar_AVX512_BS128_SHUFFLE("AVX512_BS128_SHUFFLE");


class TAVXModule_AVX2_BS256_GATHER : public TBaseCatboostModule {
public:
    TAVXModule_AVX2_BS256_GATHER(const TFullModel& model) {
        Priority = -50;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX2_BS256_GATHER, model);
        BaseName = "AVX2 GATHER Block Size 256";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX2_BS256_GATHER> AVXModuleRegistar_AVX2_BS256_GATHER("AVX2_BS256_GATHER");


class TAVXModule_AVX2_BS128_GATHER : public TBaseCatboostModule {
public:
    TAVXModule_AVX2_BS128_GATHER(const TFullModel& model) {
        Priority = -52;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX2_BS128_GATHER, model);
        BaseName = "AVX2 GATHER Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX2_BS128_GATHER> AVXModuleRegistar_AVX2_BS128_GATHER("AVX2_BS128_GATHER");



class TAVXModule_AVX512_BS512_GATHER : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS512_GATHER(const TFullModel& model) {
        Priority = -60;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS512_GATHER, model);
        BaseName = "AVX512 GATHER Block Size 512";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS512_GATHER> AVXModuleRegistar_AVX512_BS512_GATHER("AVX512_BS512_GATHER");


class TAVXModule_AVX512_BS256_GATHER : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS256_GATHER(const TFullModel& model) {
        Priority = -62;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS256_GATHER, model);
        BaseName = "AVX512 GATHER Block Size 256";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS256_GATHER> AVXModuleRegistar_AVX512_BS256_GATHER("AVX512_BS256_GATHER");


class TAVXModule_AVX512_BS128_GATHER : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS128_GATHER(const TFullModel& model) {
        Priority = -64;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS128_GATHER, model);
        BaseName = "AVX512 GATHER Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS128_GATHER> AVXModuleRegistar_AVX512_BS128_GATHER("AVX512_BS128_GATHER");


class TAVXModule_NON_SSE_FP16 : public TBaseCatboostModule {
public:
    TAVXModule_NON_SSE_FP16(const TFullModel& model) {
        Priority = -100;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::NON_SSE_FP16, model);
        BaseName = "NON SSE FP16";
    }
};

//TPerftestModuleFactory::TRegistrator<TAVXModule_NON_SSE_FP16> AVXModuleRegistar_NON_SSE_FP16("NON SSE FP16");


class TAVXModule_AVX512_BS512_FP16 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS512_FP16(const TFullModel& model) {
        Priority = -102;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS512_FP16, model);
        BaseName = "AVX512 FP16 Block Size 512";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS512_FP16> AVXModuleRegistar_AVX512_BS512_FP16("AVX512 BS512 FP16");


class TAVXModule_AVX512_BS256_FP16 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS256_FP16(const TFullModel& model) {
        Priority = -104;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS256_FP16, model);
        BaseName = "AVX512 FP16 Block Size 256";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS256_FP16> AVXModuleRegistar_AVX512_BS256_FP16("AVX512 BS256 FP16");


class TAVXModule_AVX512_BS128_FP16 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS128_FP16(const TFullModel& model) {
        Priority = -106;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS128_FP16, model);
        BaseName = "AVX512 FP16 Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS128_FP16> AVXModuleRegistar_AVX512_BS128_FP16("AVX512 BS128 FP16");


class TAVXModule_AVX512_BS512_SHUFFLE_FP16 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS512_SHUFFLE_FP16(const TFullModel& model) {
        Priority = -110;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16, model);
        BaseName = "AVX512 SHUFFLE FP16 Block Size 512";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS512_SHUFFLE_FP16> AVXModuleRegistar_AVX512_BS512_SHUFFLE_FP16("AVX512 BS512 SHUFFLE FP16");


class TAVXModule_AVX512_BS256_SHUFFLE_FP16 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS256_SHUFFLE_FP16(const TFullModel& model) {
        Priority = -112;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16, model);
        BaseName = "AVX512 SHUFFLE FP16 Block Size 256";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS256_SHUFFLE_FP16> AVXModuleRegistar_AVX512_BS256_SHUFFLE_FP16("AVX512 BS256 SHUFFLE FP16");


class TAVXModule_AVX512_BS128_SHUFFLE_FP16 : public TBaseCatboostModule {
public:
    TAVXModule_AVX512_BS128_SHUFFLE_FP16(const TFullModel& model) {
        Priority = -114;
        ModelEvaluator = NCB::NModelEvaluation::AVX::CreateAvxEvaluator(NCB::NModelEvaluation::AVX::EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16, model);
        BaseName = "AVX512 SHUFFLE FP16 Block Size 128";
    }
};

TPerftestModuleFactory::TRegistrator<TAVXModule_AVX512_BS128_SHUFFLE_FP16> AVXModuleRegistar_AVX512_BS128_SHUFFLE_FP16("AVX512 BS128 SHUFFLE FP16");


/*
class TCPUCatboostAsymmetryModule : public TBaseCatboostModule {
public:
    TCPUCatboostAsymmetryModule(const TFullModel& model) {
        CB_ENSURE(model.IsOblivious(), "model is already asymmetrical");
        TFullModel asymmetricalModel = model;
        asymmetricalModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        ModelEvaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::CPU, asymmetricalModel);
        Y_ENSURE(ModelEvaluator, "Failed to create catboost assymetrical cpu elevaluator");
        BaseName = "catboost cpu asymmetrical";
    }
};

TPerftestModuleFactory::TRegistrator<TCPUCatboostAsymmetryModule> CPUCatboostAsymmetryModuleRegistar("CPUCatboostAsymmetry");
*/

/*
class TGPUCatboostModule : public TBaseCatboostModule {
public:
    TGPUCatboostModule(const TFullModel& model) {
        ModelEvaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::GPU, model);
        Y_ENSURE(ModelEvaluator, "Failed to create catboost gpu elevaluator");
        BaseName = "catboost gpu";
    }
};

TPerftestModuleFactory::TRegistrator<TGPUCatboostModule> GPUCatboostModuleRegistar("GPUCatboostModule");
*/
