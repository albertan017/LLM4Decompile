import os
import json
import subprocess
import pandas as pd
from metrics.cal_execute_rate import execute_rate_main
import sys
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
def get_score(metric, json_file, opt, decompiler, language):
    gen_results = []
    source_codes = []
    with open(json_file) as f:
        json_obj = json.load(f)
        datas = [data for data in json_obj if data['opt'] == opt and data['language'] == language]
        if decompiler in ['ida','ghidra']:
            for data in datas:
                source_codes.append(data['func'])
                gen_result = data[f'{decompiler}_pseudo']
                old_name = gen_result.split('(')[0].split(' ')[-1]
                gen_result = gen_result.replace(old_name, data['func_name'])
                gen_results.append(gen_result)  
        else:
            for data in datas:
                fileName = str(data['index']) + '_' + opt
                jsonName = json_file.split('/')[-1].replace('.json','')
                filePath = f'./model_outputs/{jsonName}/{decompiler}/{opt}/{fileName}.{data["language"]}'
                if not os.path.exists(filePath):
                    code = ''
                else:
                    with open(filePath) as file:
                        code = file.read()
                gen_results.append(code)
                source_codes.append(data['func'])     

        for idx in range(len(gen_results)):
            one = gen_results[idx]
            func_name = one.split('(')[0].split(' ')[-1].strip()
            if func_name.strip() and func_name[0:2] == '**':
                func_name = func_name[2:]
            elif func_name.strip() and func_name[0] == '*':
                func_name = func_name[1:]
            
            one = one.replace(func_name, datas[idx]["func_name"])
            gen_results[idx] = one
        else:
            return execute_rate_main(datas,gen_results,language=language)


def eval(json_file, lan, decompilers, metrics):
    optimal_level = ['O0', 'O1', 'O2', 'O3']
    for opt in optimal_level:
        result = {'decompiler': decompilers}
        output_name = json_file.split('/')[-1].split('.')[0] + '_' + lan + '_' + opt + '.csv'
        for metric in metrics:
            scores = [get_score(metric, json_file, opt, decompiler, lan) for decompiler in decompilers]
            result[metric] = scores
        df = pd.DataFrame(result)
        df.to_csv(f'{current_dir}/results/{output_name}', index = False)    


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--json_file", type=str, default='./data/mbpp_normsrcpseudo.json')
    arg_parser.add_argument("--language", type=str, default='c')
    arg_parser.add_argument("--decompilers", type=str, default='gpt-5-mini-name7,idioms,lmdc6.7,pseudo2normFinal-Debug,pseudo2normFinal_RL-Debug,pseudo2norm_Final+norm2codeFinal-Debug-11000,pseudo2norm_RLFinal+norm2codeFinal-Debug-11000,pseudo2code_Final-3200-Debug')
    arg_parser.add_argument("--metrics", type=str, default='exe_rate')
    args = arg_parser.parse_args()
    decompilers = args.decompilers.split(",")
    metrics = args.metrics.split(",")
    eval(args.json_file, args.language, decompilers, metrics)