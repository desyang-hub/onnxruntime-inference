/**
 * @FilePath     : /onnxruntime-inference/include/arch/arch.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-02 17:17:04
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 18:55:58
**/
#pragma once

#include <string>

#ifdef _WIN32
    #include <windows.h>

    // UTF-8 string → wstring
    inline std::wstring utf8_to_wide(const std::string& utf8) {
        if (utf8.empty()) return {};
        int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), 
                                    static_cast<int>(utf8.size()), nullptr, 0);
        std::wstring result(len, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), 
                            static_cast<int>(utf8.size()), &result[0], len);
        return result;
    }
    #define MODEL_PATH(model_path) utf8_to_wide(model_path)

#else

    #define MODEL_PATH(model_path) (model_path)

#endif