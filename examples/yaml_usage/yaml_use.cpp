/**
 * @FilePath     : /onnxruntime-infer/examples/yaml_usage/yaml_use.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 21:06:04
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 21:07:08
**/
#include <yaml-cpp/yaml.h>
#include <iostream>

int main(int argc, char const *argv[])
{
    
    auto config = YAML::LoadFile("config/config.yaml");
    std::cout << config << std::endl;




    return 0;
}
