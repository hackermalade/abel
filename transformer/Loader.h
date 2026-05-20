#pragma once

#include "Model.h"
#include <string>

class ModelLoader {
public:
    ModelLoader(const std::string& path);
    NyxTransformer load();

private:
    std::string path_;
};
