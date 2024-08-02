## Updates
* [2024-06-20]: Upload ``decompile-eval-executable-gcc-ghidra.json``. The source codes are compiled into executable binaries and disassembled by Ghidra into pseudo-code.
* [2024-05-16]: Please use ``decompile-eval-executable-gcc-obj.json``. The source codes are compiled into executable binaries and disassembled into assembly instructions.

There are 164*4 (O0, O1, O2, O3) samples, each with five keys:

*   ``task_id``: indicates the ID of the problem.
*   ``type``: the optimization stage, is one of [O0, O1, O2, O3].
*   ``c_func``: C solution for HumanEval problem. 
*   ``c_test``: C test assertions.
*   ``input_asm_prompt``: assembly instructions with prompts (or Ghidra results), can be derived as in our preprocessing example for [direct decompile](https://github.com/albertan017/LLM4Decompile#quick-start) or [refine Ghidra](https://github.com/albertan017/LLM4Decompile/blob/main/ghidra/README.md#quick-start).

Programming languages are highly structured and logical, insensitive to the naming of functions and variables, yet very sensitive to the flow of data and logic. Changing variable or function names does not affect the meaning of a program, but a single logical error can alter its entire function and purpose.
As illustrated in Figure, the use of BLEU and ES in evaluating code similarity is problematic. 
For $src_1$, the variation from the original $src$ is confined to variable $num$'s type conversion, which leads to high BLEU and ES scores. However, this alteration completely changes the intent of the code. Similarly, $src_2$ achieves high BLEU and ES scores, yet the semantics of the function are lost. Conversely, $src_3$ undergoes normalization of function and variable names, causing no semantic shift yet scoring zero in BLEU against the original code. The example of $src_4$ is more extreme: if the program logic is broken down into multiple lines, the ES drops to 41.4\%, falsely indicating a low similarity. However, during compilation, names are typically standardized by the compiler, and source code is often broken down into basic operations depending on optimization. For this reason, the ability to recompile and execute the code is far more indicative than N-gram or edit similarity for evaluating decompilation efficacy.

<p align="center">
<img src="https://github.com/albertan017/LLM4Decompile/blob/main/samples/case.png" alt="image" width="300" height="auto">
</p>

To address the gap in decompilation assessment, we introduce Decompile-Eval to evaluate the re-executability of decompilation systems. This benchmark is derived from HumanEval, which is the leading benchmark for code generation assessment and includes 164 programming challenges with accompanying Python solutions and assertions. We converted these Python solutions and assertions into C, making sure that they compile with the GCC compiler using standard C libraries and pass all the original assertions. In our [evaluation process](https://github.com/albertan017/LLM4Decompile/blob/main/samples/case.png), the C source code is first compiled into a binary, then disassembled into assembly code, and finally fed into the decompilation system to be reconstructed back into C source code. This regenerated C code is combined with the original assertions to check if it can successfully execute and pass those assertions. 

