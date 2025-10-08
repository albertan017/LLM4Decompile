import json
import argparse
import os
from metrics.R2I.run import run_r2i
from inf_type import process_one
import warnings
import sys
import shutil
current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--json_file",type=str,default='./data/humaneval_normsrcpseudo.json')
    arg_parser.add_argument("--decompilers", type=str, default='gpt-5-mini-name7,idioms,lmdc6.7,pseudo2norm_RLFinal+norm2codeFinal-Debug-11000')                  
    arg_parser.add_argument("--output_path", type=str, default='./result/humaneval_normsrcpseudo')
    arg_parser.add_argument("--generator", default="./psychec/psychecgen", help="Path to your generator executable")
    arg_parser.add_argument("--solver", default="./psychec/psychecsolver-exe", help="Name of your solver (for `stack exec …`)") 
    args = arg_parser.parse_args()
    
    decompilers = args.decompilers.split(",")
    if len(decompilers) < 2:
        warnings.warn("To calculate the R2I metric, at least two decompilers are needed")
        sys.exit(1)

    dataset = args.json_file.split('/')[-1].replace('.json','')
    with open(args.json_file) as f:
        datas = json.load(f)
        datas = [data for data in datas if data['language'] == 'c']

    shutil.rmtree(f'metrics/R2I/dataset/test')
    for decompiler in decompilers:
        os.makedirs(f'metrics/R2I/dataset/test/{decompiler}/c', exist_ok=True)
        os.makedirs(f'metrics/R2I/dataset/test/{decompiler}/json', exist_ok=True)
        os.makedirs(f'metrics/R2I/dataset/test/{decompiler}/syntax_correction', exist_ok=True)
    opts = ["O0", "O1", "O2", "O3"]
    for opt in opts:
        names = []
        i = 0
        for data in datas:
            if data['opt'] == opt:
                func_name = data['func_name']
                index = data['index']
                names.append(func_name)
                for decompiler in decompilers:
                    if not os.path.exists(f'./model_outputs/{dataset}/{decompiler}/{opt}/{index}_{opt}.c'):
                        prediction = ''
                    else:   
                        with open(f'./model_outputs/{dataset}/{decompiler}/{opt}/{index}_{opt}.c') as f:
                            prediction = f.read().replace('__fastcall', '')
                    old_name = prediction.split('(')[0].split(' ')[-1].strip()
                    if old_name.strip() and old_name[0:2] == '**':
                        old_name = old_name[2:]
                    elif old_name.strip() and old_name[0] == '*':
                        old_name = old_name[1:]
                    prediction = prediction.replace(old_name, func_name)
                    if os.path.exists(f'./headers/{dataset}/{decompiler}/{opt}/{index}_{opt}.h'):
                        with open(f'./headers/{dataset}/{decompiler}/{opt}/{index}_{opt}.h') as f:
                            header = f.read()
                    else:
                        header = process_one(prediction, args.generator, args.solver)
                    if header:
                        prediction = header + prediction
                    prediction_json = {}
                    prediction_json['isStripped'] = True 
                    prediction_json['decompilerName'] = decompiler
                    prediction_json['compilerName'] = 'gcc'
                    prediction_json['optLevel'] = "-O"
                    prediction_json['funcInfo'] = [{"funcName":func_name,"decompiledFuncCode":prediction}]
                    with open(f'metrics/R2I/dataset/test/{decompiler}/c/file{i}.c','w') as f:
                        f.write(prediction)
                    with open(f'metrics/R2I/dataset/test/{decompiler}/json/file{i}.json','w') as f:
                        json.dump(prediction_json, f, indent=4, ensure_ascii=False)
                    with open(f'metrics/R2I/dataset/test/{decompiler}/syntax_correction/file{i}.c','w') as f:
                        f.write(prediction)
                i += 1            

        scores = run_r2i(decompilers, names)
        os.chdir(current_dir)
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, args.output_path.split('/')[-1]+'_results.txt'), 'a') as f:
            for key in scores:
                f.write(f'r2i {opt} {key}: {scores[key]*100:.2f}, ')
            f.write("\n")