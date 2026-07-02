/**
 * @FilePath     : /onnxruntime-infer/include/registry.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 16:40:27
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 16:55:32
**/
#include <functional>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

#include "preprocess/Preprocessor.h"
#include "backend/InferenceBackend.h"

// 通用工厂模板
template <typename Interface>
class Factory {
    using Creator = std::function<std::unique_ptr<Interface>(const YAML::Node&)>;
    std::unordered_map<std::string, Creator> creators_;
public:
    static Factory& instance() {
        static Factory f;
        return f;
    }
    
    void register_type(const std::string& name, Creator creator) {
        creators_[name] = std::move(creator);
    }
    
    std::unique_ptr<Interface> create(const std::string& name, 
                                      const YAML::Node& params) {
        auto it = creators_.find(name);
        if (it == creators_.end())
            throw std::runtime_error("Unknown type: " + name);
        return it->second(params);
    }

    static std::unique_ptr<Interface> Create(const std::string& name, 
        const YAML::Node& params) {
        return instance().create(name, params);
    }
};

// ⭐ 自注册宏：放在 .cpp 文件顶部即可自动注册
#define REGISTER_PREPROCESSOR(name, cls) \
    static bool _reg_##cls = []{ \
        Factory<Preprocessor>::instance().register_type( \
            name, [](const YAML::Node& p){ \
                return std::make_unique<cls>(p); }); \
        return true; \
    }();

#define REGISTER_POSTPROCESSOR(name, cls) \
    static bool _reg_##cls = []{ \
        Factory<Postprocessor>::instance().register_type( \
            name, [](const YAML::Node& p){ \
                return std::make_unique<cls>(p); }); \
        return true; \
    }();