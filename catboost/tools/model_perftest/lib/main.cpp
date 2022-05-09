#include "perftest_module.h"
#include <catboost/private/libs/algo/features_data_helpers.h>

#include <catboost/libs/data/load_data.h>

#include <catboost/libs/logging/logging.h>

#include <library/cpp/json/json_value.h>
#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/threading/future/async.h>

#include <util/system/info.h>
#include <util/generic/algorithm.h>
#include <util/system/hp_timer.h>
#include <util/thread/pool.h>

#include <ios>
#include <fstream>


struct TCMDOptions {
    TString PoolPath;
    TString CdPath;
    TString ModelPath;
    size_t BlockSize = Max<size_t>();
    size_t RepetitionCount = 1;
    bool SingleBlock = false;
    int ThreadCount = 1;
};

struct TTimingResult {
    TVector<double> Times;

    void Add(double val) {
        Times.push_back(val);
    }

    double Min() const {
        return *MinElement(Times.begin(), Times.end());
    }

    double Max() const {
        return *MaxElement(Times.begin(), Times.end());
    }

    double Mean() const {
        double sum = 0;
        for (auto t : Times) {
            sum += t;
        }
        return sum / Times.size();
    }

    void Output(const TTimingResult* ref = nullptr) const {
        /*
        auto myMin = Min();
        CATBOOST_INFO_LOG << "min:\t" << myMin;
        if (ref) {
            CATBOOST_INFO_LOG << "\t" << myMin/ref->Min();
        }
        CATBOOST_INFO_LOG << Endl;

        auto myMax = Max();
        CATBOOST_INFO_LOG << "max:\t" << myMax;
        if (ref) {
            CATBOOST_INFO_LOG << "\t" << myMax/ref->Max();
        }
        CATBOOST_INFO_LOG << Endl;
        */

        auto mymean = Mean();
        CATBOOST_INFO_LOG << "mean:\t" << mymean;
        if (ref) {
            CATBOOST_INFO_LOG << "\t" << mymean / ref->Mean();
        }
        CATBOOST_INFO_LOG << Endl;
    }

    NJson::TJsonValue GetJsonValue() const {
        NJson::TJsonValue result;
        result["min"] = Min();
        result["max"] = Max();
        result["mean"] = Mean();
        return result;
    }
};

struct TResults {
    TMap<std::pair<int, TString>, THolder<TTimingResult>> Results;
    std::pair<int, TString> BaseResultKey;
    TAdaptiveLock Lock;

    void UpdateResult(int priority, const TString& name, double time) {
        with_lock(Lock) {
            BaseResultKey = min(BaseResultKey, {-priority, name});
            auto& value = Results[{-priority, name}];
            if (!value) {
                value = MakeHolder<TTimingResult>();
            }
            value->Add(time);
        }
    }

    void OutputResults() const {
        if (!Results) {
            return;
        }
        NJson::TJsonValue jsonValue;
        const TTimingResult* refTimingResult = nullptr;
        CATBOOST_INFO_LOG << "name\tvalue\tdiff" << Endl;

        refTimingResult = Results.at(BaseResultKey).Get();
        for (const auto& [key, value] : Results) {
            CATBOOST_INFO_LOG << key.second << "\t" << Endl;
            value->Output(refTimingResult);
            jsonValue[key.second] = value->GetJsonValue();
        }
        TFileOutput resultsFile("results.json");
        resultsFile << jsonValue.GetStringRobust();
    }
};

struct TCanonData {
    static constexpr float Epsilon = 1e-6;
    TCanonData(size_t blockCount) {
        Results.resize(blockCount);
    }

    void CheckOrSet(const TString& name, size_t blockId, const TVector<double>& blockResult) {
        if (Results[blockId].empty()) {
            Results[blockId] = blockResult;
            RefName = name;
        }
        const auto& ref = Results[blockId];
        CB_ENSURE(blockResult.size() == ref.size());
        for (size_t i = 0; i < blockResult.size(); ++i) {
            if(abs(blockResult[i] - ref[i]) > Epsilon) {
                Cerr << LabeledDump(name, RefName, blockId, i, blockResult[i], ref[i], blockResult[i] - ref[i]) << Endl;
            }
        }
    }
    TVector<TVector<double>> Results;
    TString RefName;
};


TVector<bool> GetFeaturesUsedInModel(const TFullModel& model) {
    TVector<bool> result(model.ModelTrees->GetFlatFeatureVectorExpectedSize(), false);

    for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
        if (floatFeature.UsedInModel()) {
            result[floatFeature.Position.FlatIndex] = true;
        }
    }

    for (const auto& catFeature : model.ModelTrees->GetCatFeatures()) {
        if (catFeature.UsedInModel()) {
            result[catFeature.Position.FlatIndex] = true;
        }
    }

    return result;
}


int DoMain(int argc, char** argv) {
    TCMDOptions options;
    auto parser = NLastGetopt::TOpts();
    parser.AddLongOption('f', "pool-path")
        .StoreResult(&options.PoolPath)
        .Required();
    parser.AddLongOption("cd")
        .StoreResult(&options.CdPath)
        .Required();
    parser.AddLongOption('m', "model-path")
        .StoreResult(&options.ModelPath)
        .Required();
    parser.AddLongOption("block-size")
        .StoreResult(&options.BlockSize)
        .Optional();
    parser.AddLongOption("repetitions")
        .StoreResult(&options.RepetitionCount)
        .Optional();
    parser.AddLongOption("single-block")
        .StoreTrue(&options.SingleBlock);
    parser.AddLongOption("threads")
        .StoreResult(&options.ThreadCount)
        .Optional();

    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    TFullModel model = ReadModel(options.ModelPath);

    TVector<bool> featureUsedInModel = GetFeaturesUsedInModel(model);

    if (options.ThreadCount > 1) {
        NPar::LocalExecutor().RunAdditionalThreads(options.ThreadCount - 1);
    }

    NCatboostOptions::TColumnarPoolFormatParams columnarPoolFormatParams;
    columnarPoolFormatParams.CdFilePath = NCB::TPathWithScheme(options.CdPath, "dsv");
    NCB::TDataProviderPtr dataset = NCB::ReadDataset(
        /*taskType*/Nothing(),
        NCB::TPathWithScheme(options.PoolPath, "dsv"),
        NCB::TPathWithScheme(),
        NCB::TPathWithScheme(),
        NCB::TPathWithScheme(),
        NCB::TPathWithScheme(),
        NCB::TPathWithScheme(),
        NCB::TPathWithScheme(),
        columnarPoolFormatParams,
        TVector<ui32>(),
        NCB::EObjectsOrder::Undefined,
        NSystemInfo::CachedNumberOfCpus(),
        true,
        /*forceUnitAutoPairWeights*/ false,
        /*classNames*/ Nothing()
    );

    THolder<NCB::IFeaturesBlockIterator> featuresBlockIterator
        = NCB::CreateFeaturesBlockIterator(model, *(dataset->ObjectsData), 0, dataset->GetObjectCount());

    NCB::TRawFeaturesBlockIterator* rawFeaturesBlockIterator
        = dynamic_cast<NCB::TRawFeaturesBlockIterator*>(featuresBlockIterator.Get());

    CB_ENSURE(rawFeaturesBlockIterator, "Not supported for quantized pools");

    rawFeaturesBlockIterator->NextBlock((size_t)dataset->GetObjectCount());

    const auto& featuresLayout = *dataset->MetaInfo.FeaturesLayout;

    auto getFeatureDataBeginPtr = [&] (size_t flatFeatureIdx) -> const float* {
        auto featureType = featuresLayout.GetExternalFeatureType(flatFeatureIdx);
        switch (featureType) {
            case EFeatureType::Float:
                return rawFeaturesBlockIterator->GetFloatValues()[flatFeatureIdx].data();
            case EFeatureType::Categorical:
                return (const float*)rawFeaturesBlockIterator->GetCatValues()[flatFeatureIdx].data();
            default:
                CB_ENSURE(false, "Unsupported column type :" << featureType);
        }
    };


    options.BlockSize = Min(options.BlockSize, (size_t)dataset->GetObjectCount());
    Y_ENSURE(options.BlockSize > 0, "Empty pool");
    const size_t docsCount = dataset->GetObjectCount();
    const size_t blockCount = (options.SingleBlock ? 1 : (docsCount) / options.BlockSize);
    const size_t factorsCount = featureUsedInModel.size();

    CATBOOST_DEBUG_LOG << "Blocks count: " << blockCount << " block size: " << options.BlockSize << Endl;

    TVector<float> ignoredFeatureData(options.BlockSize);
    TVector<TVector<TVector<float>>> nonTransposedPool(blockCount);
    TVector<TVector<TConstArrayRef<float>>> nonTranspFactorsRef(blockCount);
    TVector<TVector<TConstArrayRef<float>>> transpFactorsRef(blockCount);

    for(size_t blockId = 0; blockId < blockCount; ++blockId) {
        const size_t blockStart = options.BlockSize * blockId;
        const size_t docsInCurrBlock = Min<size_t>(options.BlockSize, docsCount - options.BlockSize * blockId);
        CB_ENSURE(docsInCurrBlock >= 0);
        transpFactorsRef[blockId].resize(factorsCount);
        for (size_t i = 0; i < factorsCount; ++i) {
            if (featureUsedInModel[i]) {
                transpFactorsRef[blockId][i] = MakeArrayRef<const float>(
                    getFeatureDataBeginPtr(i) + blockStart,
                    docsInCurrBlock);
            } else {
                transpFactorsRef[blockId][i] = MakeArrayRef<const float>(
                    ignoredFeatureData.data(),
                    docsInCurrBlock);
            }
        }

        nonTransposedPool[blockId].resize(docsInCurrBlock);
        nonTranspFactorsRef[blockId].resize(docsInCurrBlock);
        for (size_t docId = 0; docId < docsInCurrBlock; ++docId) {
            auto &docFacs = nonTransposedPool[blockId][docId];
            docFacs.resize(factorsCount);
            for (size_t featureId = 0; featureId < factorsCount; ++featureId) {
                if (featureUsedInModel[featureId]) {
                    docFacs[featureId] = getFeatureDataBeginPtr(featureId)[blockStart + docId];
                } else {
                    docFacs[featureId] = 0.0f;
                }
            }
            nonTranspFactorsRef[blockId][docId] = MakeArrayRef(docFacs);
        }
    }
    TVector<double> results_vec(options.BlockSize);
    TResults results;
    TCanonData canonData(blockCount);
    TCanonData canonDataFp16(blockCount);
    TVector<THolder<IPerftestModule>> modules;
    TSet<TPerftestModuleFactory::TKey> allRegisteredKeys;
    TPerftestModuleFactory::GetRegisteredKeys(allRegisteredKeys);
    for (const auto& key : allRegisteredKeys) {
        try {
            auto product = TPerftestModuleFactory::Construct(key, model);
            if (!product) {
                continue;
            }
            modules.emplace_back(std::move(product));
        } catch (yexception& e) {
            Cerr << "Failed to construct module " << key << " got error: " << e.what() << Endl;
        } catch (...) {
            Cerr << "Failed to construct module " << key << " got error: " << CurrentExceptionMessage() << Endl;
        }
    }

    for (auto& module : modules) {
        if (module->SupportsLayout(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst)) {
            auto& cn = module->GetName(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst).find("FP16") == TString::npos ? canonData : canonDataFp16;
            for (size_t blockId = 0; blockId < blockCount; ++blockId) {
                cn.CheckOrSet(
                    module->GetName(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst),
                    blockId,
                    module->GetApplyResult(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst, nonTranspFactorsRef[blockId])
                );
            }
        }
        if (module->SupportsLayout(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst)) {
            auto& cn = module->GetName(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst).find("FP16") == TString::npos ? canonData : canonDataFp16;
            for (size_t blockId = 0; blockId < blockCount; ++blockId) {
                cn.CheckOrSet(
                    module->GetName(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst),
                    blockId,
                    module->GetApplyResult(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst, transpFactorsRef[blockId])
                );
            }
        }
    }

    /*std::ofstream canon_output("canon_data");
    canon_output << std::fixed << std::setprecision(100);
    for (const auto& x : canonData.Results) {
        for (const auto& y : x) {
            canon_output << y << '\t';
        }
    }
    canon_output << '\n';
    for (const auto& x : canonDataFp16.Results) {
        for (const auto& y : x) {
            canon_output << y << '\t';
        }
    }
    canon_output << '\n';*/

    for (size_t i = 0; i < options.RepetitionCount; ++i) {
        for (auto& module : modules) {
            if (module->SupportsLayout(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst)) {
                if (options.ThreadCount == 1) {
                    for (size_t blockId = 0; blockId < blockCount; ++blockId) {
                        results.UpdateResult(
                            module->GetComparisonPriority(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst),
                            module->GetName(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst),
                            module->Do(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst, nonTranspFactorsRef[blockId])
                        );
                    }
                } else {
                    THPTimer timer;
                    NPar::LocalExecutor().ExecRangeWithThrow([&](int blockId) {
                        module->Do(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst, nonTranspFactorsRef[blockId]);
                    }, 0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);
                    results.UpdateResult(
                        module->GetComparisonPriority(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst),
                        module->GetName(IPerftestModule::EPerftestModuleDataLayout::ObjectsFirst),
                        timer.Passed()
                    );
                }
            }
            if (module->SupportsLayout(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst)) {
                if (options.ThreadCount == 1) {
                    for (size_t blockId = 0; blockId < blockCount; ++blockId) {
                        results.UpdateResult(
                            module->GetComparisonPriority(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst),
                            module->GetName(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst),
                            module->Do(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst, transpFactorsRef[blockId])
                        );
                    }
                } else {
                    THPTimer timer;
                    NPar::LocalExecutor().ExecRangeWithThrow([&](int blockId) {
                        module->Do(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst, transpFactorsRef[blockId]);
                    }, 0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);
                    results.UpdateResult(
                        module->GetComparisonPriority(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst),
                        module->GetName(IPerftestModule::EPerftestModuleDataLayout::FeaturesFirst),
                        timer.Passed()
                    );
                }
            }
        }
    }
    results.OutputResults();

    return 0;
}

int main(int argc, char** argv) {
    try {
        auto queue = CreateThreadPool(1);

        NThreading::TFuture<void> future = NThreading::Async(
            [=](){
                TSetLoggingVerbose inThisScope;
                DoMain(argc, argv);
            },
            *queue
        );
        future.GetValueSync();
    } catch (...) {
        Cerr << CurrentExceptionMessage() << Endl;
        return -1;
    }
    return 0;
}
