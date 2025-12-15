const fs = require('fs');
const path = require('path');

// 读取你的 C++ 文件内容
const sourceCode = fs.readFileSync(path.join(__dirname, 'wcn_math_wasm.c'), 'utf8');

// 正则表达式匹配 WCN_WASM_EXPORT 宏
// 解释:
// WCN_WASM_EXPORT  -> 匹配宏
// \s+              -> 匹配宏后面的空格
// [\w\s\*]+        -> 匹配返回类型 (包括 void, vec2_t*, 空格等)
// \s+              -> 匹配类型和函数名之间的空格
// (\w+)            -> 【捕获组1】匹配函数名
// \s*\(            -> 匹配函数名后面的左括号 (可能有空格)
const regex = /WCN_WASM_EXPORT\s+[\w\s\*]+\s+(\w+)\s*\(/g;

const functions = [];
let match;

while ((match = regex.exec(sourceCode)) !== null) {
    // match[1] 是捕获的函数名
    functions.push("_" + match[1]);
}

// 输出 JSON 数组格式，方便直接复制到 JS 代码中使用
console.log(JSON.stringify(functions, null, 2));

// 或者输出为 EMSCRIPTEN 需要的命令行参数格式
// console.log(functions.map(f => `_${f}`).join(','));