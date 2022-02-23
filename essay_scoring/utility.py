import numpy as np

asap_score_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}

def get_normalized_score(actual_score, low, high):
    return (actual_score - low)/(high - low)

def get_actual_score(normalized_score, prompt_id):
    low, high = asap_score_ranges.get(prompt_id)
    actual_score = (normalized_score * (high - low)) + low
    return np.around(actual_score).astype(int) 

def calculate_actual_score(normalized_score_list, prompt_id_list):
    assert len(normalized_score_list)==len(prompt_id_list), "length mismatch between normalized score and prompt_id"
    res = []
    for norm_score, prompt_id in zip(normalized_score_list, prompt_id_list):
        res.append(get_actual_score(norm_score, prompt_id))
    return res

def load_txt(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as dfile:
        for line in dfile.readlines():
            temp = line.strip()
            if temp=='' or len(temp)==0:
                continue
            res.append(temp)
    return res