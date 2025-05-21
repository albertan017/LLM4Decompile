import editdistance
def compute_ES(target, prediction):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_str = '\n'.join(prediction_lines)
    ES_score=1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
    
    return ES_score

if __name__ == '__main__':
    import json
    import os
    import numpy as np
    for dataset_name in ['humaneval','mbpp']:
        # raw data
        with open(f'./data/{dataset_name}-decompile.json', 'r') as f:
            data = json.load(f)
        # decompiled results
        decompile_results = {}
        for root, dirs, files in os.walk(f'./data/{dataset_name}'):
            for name in files:
                if name.endswith('.c') or name.endswith('.cpp'):
                    full_path = os.path.join(root, name)
                    with open(full_path, 'r') as f:
                        prediction = f.read().strip()
                        index = int(name.split('_')[0])
                    decompile_results[index] = prediction
                
        es_all = []
        for item in data:
            es = compute_ES(item['func'], decompile_results[item['index']])
            es_all.append(es)
        print(dataset_name)
        print(np.average(es_all))
