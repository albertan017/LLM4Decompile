import openai
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

scores_tmp = {
    "Code Readability Assessment": {
        "score": 1,
        "rationale": "<string>"
    }
}

def eval_func(write_path, input_prompt, api_key, max_retries=5, initial_delay=1):
    openai.base_url = "https://api5.xhub.chat/v1/"
    openai.api_key = api_key
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-5-mini",#"gpt-4o",##"gpt-4o-mini",#"gpt-5",
                messages=[{"role": "user", "content": input_prompt}],
                max_tokens=8192,
                temperature=0,
            )
            answer = response.choices[0].message.content
            if not isinstance(answer, str):
                answer = response.choices[0].message.content[1]['text']
            
            try:
                txt = answer.strip()
                txt = txt.replace("```json","").replace("```", "").strip()
                score = json.loads(txt)
                for score_name in scores_tmp:
                    score_one = score[score_name]
            except Exception as e:
                if "Expecting ',' delimiter" in str(e):
                    try:
                        txt = answer.strip()
                        txt = txt.replace("```json","").replace("```", "").strip()+'}'
                        score = json.loads(txt)
                        for score_name in scores_tmp:
                            score_one = score[score_name]
                    except Exception as e2:
                        raise ValueError()
                elif "Invalid \\escape:" in str(e):
                    try:
                        txt = answer.strip()
                        txt = txt.replace("```json","").replace("```", "").strip().replace('\\','')
                        score = json.loads(txt)
                        for score_name in scores_tmp:
                            score_one = score[score_name]
                    except Exception as e2:
                        raise ValueError()
                else:
                    raise ValueError()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                # delay *= 2 
                # print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                answer = f"# Error during judge: {str(e)}"
                print(answer)

    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    with open(write_path, 'w') as f:
        f.write(txt)


def eval_score(json_file, decompiler, opt):
    with open(json_file) as f:
        datas = json.load(f)
        datas = [data for data in datas if data['opt'] == opt]

    scores = {score_name: [] for score_name in scores_tmp}
    for data in datas:
        opt = data['opt']
        language = data['language']
        file_name = os.path.basename(json_file).replace('.json', '')
        output_name = str(data['index']) + '_' + opt
        score_path = f'{current_dir}/judge_outputs/{file_name}/{decompiler}/{opt}/{output_name}.{language}'
        try:
            with open(score_path, 'r') as f:
                txt = f.read().strip()
                txt = txt.replace("```json","").replace("```", "").strip()
                score = json.loads(txt)
                for score_name in scores:
                    score_one = score.get(score_name, {"score":1})
                    scores[score_name].append(int(score_one["score"]))
        except Exception as e:
            if "Expecting ',' delimiter" in str(e):
                try:
                    with open(score_path, 'r') as f:
                        txt = f.read().strip()
                        txt = txt.replace("```json","").replace("```", "").strip()+'}'
                        score = json.loads(txt)
                        for score_name in scores:
                            score_one = score.get(score_name, {"score":1})
                            scores[score_name].append(int(score_one["score"]))
                except Exception as e2:
                    print(f"Error loading score for {score_path}: {str(e2)}")
                    for score_name in scores:
                        scores[score_name].append(1)
            elif "Invalid \\escape:" in str(e):
                try:
                    with open(score_path, 'r') as f:
                        txt = f.read().strip()
                        txt = txt.replace("```json","").replace("```", "").strip().replace('\\','')
                        score = json.loads(txt)
                        for score_name in scores:
                            score_one = score.get(score_name, {"score":1})
                            scores[score_name].append(int(score_one["score"]))
                except Exception as e2:
                    print(f"Error loading score for {score_path}: {str(e2)}")
                    for score_name in scores:
                        scores[score_name].append(1)
            else:
                print(f"Error loading score for {score_path}: {str(e)}")
                for score_name in scores:
                    scores[score_name].append(1)
    return scores

def eval_funcs(json_file, decompiler, prompt, opt, api_key):
    tasks = []
    with open(json_file) as f:
        datas = json.load(f)
        datas = [data for data in datas if data['opt'] == opt]

    with ThreadPoolExecutor(max_workers=64) as executor:  # 可根据实际情况调整线程数
        for data in datas:
            opt = data['opt']
            func = data['func']
            func_name = func.split('(')[0].split(' ')[-1].strip()
            if func_name.strip() and func_name[0:2] == '**':
                func_name = func_name[2:]
            elif func_name.strip() and func_name[0] == '*':
                func_name = func_name[1:]
            
            func = func.replace(func_name, 'func0')
            language = data['language']
            file_name = os.path.basename(json_file).replace('.json', '')
            output_name = str(data['index']) + '_' + opt
            decompile_path = f'./model_outputs/{file_name}/{decompiler}/{opt}/{output_name}.{language}'
            try:
                with open(decompile_path, 'r') as f:
                    decompile_result = f.read().strip()
            except:
                decompile_result = 'decompile error'
                print(decompile_result)
            input_prompt = prompt.replace('[SRC]', func).replace('[DSRC]', decompile_result)
            write_path = f'{current_dir}/judge_outputs/{file_name}/{decompiler}/{opt}/{output_name}.{language}'
            tasks.append(executor.submit(eval_func, write_path, input_prompt, api_key))

        # 可选：等所有任务完成后进行处理或打印进度
        for future in as_completed(tasks):
            future.result()  # 捕获异常，确保所有任务完成

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--json_file",type=str,default='./data/humaneval_normsrcpseudo.json')
    arg_parser.add_argument("--prompt",type=str,default='template.txt')
    arg_parser.add_argument("--decompilers",type=str,default='gpt-5-mini-name7,idioms,lmdc6.7,pseudo2normFinal-Debug,pseudo2normFinal_RL-Debug,pseudo2norm_Final+norm2codeFinal-Debug-11000,pseudo2norm_RLFinal+norm2codeFinal-Debug-11000,pseudo2code_Final-3200')
    arg_parser.add_argument("--cal_score",type=int,default=1)
    arg_parser.add_argument("--opt",type=str,default='O0')
    arg_parser.add_argument("--api_key",type=str)
    args = arg_parser.parse_args()
    with open(args.prompt, 'r') as f:
        prompt = f.read()
    decompilers = args.decompilers.split(",")
    opt = args.opt
    if args.cal_score == 0:
        for decompiler in decompilers:
            eval_funcs(args.json_file, decompiler, prompt)
    if args.cal_score == 1:
        scores = {}
        scores_str = f'{opt}:\n'
        file_name = os.path.basename(args.json_file).replace('.json', '')
        for decompiler in decompilers:
            eval_funcs(args.json_file, decompiler, prompt, opt, args.api_key)
            scores[decompiler] = eval_score(args.json_file, decompiler, opt)
            score_string = f'{decompiler}:'
            for socre_key in scores[decompiler]:
                score_list = scores[decompiler][socre_key]
                score_string += f'{socre_key}:{sum(score_list)/len(score_list):.2f};'
            print(score_string)
            scores_str += score_string+'\n'
        with open(f'{current_dir}/{file_name}_gpt5minijudge_src.txt', 'w') as f:
            f.write(scores_str)
    if args.cal_score == 2:
        scores = {}
        scores_str = ''
        file_name = os.path.basename(args.json_file).replace('.json', '')
        for decompiler in decompilers:
            scores[decompiler] = eval_score(args.json_file, decompiler, opt)
            score_string = f'{decompiler}:'
            for socre_key in scores[decompiler]:
                score_list = scores[decompiler][socre_key]
                score_string += f'{socre_key}:{sum(score_list)/len(score_list):.2f};'
            print(score_string)
            scores_str += score_string+'\n'
        with open(f'{current_dir}/{file_name}_gpt5minijudge.txt', 'w') as f:
            f.write(scores_str)
if __name__ == '__main__':
    main()
