#!/usr/bin/env python3
"""
从 WCN_WASM.h 中提取所有导出的函数名，返回逗号分隔的列表
"""

import re
import sys
import os

def extract_export_functions():
    """提取WCN_WASM.h中的导出函数"""
    header_file = 'include/WCN/WCN_WASM.h'
    
    if not os.path.exists(header_file):
        print(f"错误：找不到文件 {header_file}", file=sys.stderr)
        return ""
    
    try:
        with open(header_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件失败: {e}", file=sys.stderr)
        return ""
    
    # 匹配 WCN_WASM_EXPORT 函数定义的模式
    pattern = r'WCN_WASM_EXPORT\s+(?:[^;]+\s+)?(\w+)\s*\([^)]*\)\s*;'
    
    matches = re.findall(pattern, content)
    
    # 过滤和去重
    functions = []
    for func in matches:
        if func not in ['WCN_Context', 'WGPUTextureFormat'] and func not in functions:
            functions.append(func)
    
    # 转换为导出的函数名格式 (添加下划线前缀)
    exported_functions = []
    for func in functions:
        if func.startswith('wcn_'):
            exported_functions.append(f'_{func}')
        elif func.startswith('_wcn_'):
            exported_functions.append(func)
        else:
            exported_functions.append(f'_{func}')
    
    # 添加基本内存管理函数
    exported_functions.extend(['_malloc', '_free'])
    
    return ','.join(exported_functions)

if __name__ == '__main__':
    exports = extract_export_functions()
    if exports:
        print(exports)
    else:
        print("", file=sys.stderr)
        sys.exit(1)