#scp + 5 klas
import ast
import numpy as np

SUPERCLASSES = {
    "NORM": ["NORM"],
    "MI": ["MI"],
    "STTC": ["STTC"],
    "CD": ["CD"],
    "HYP": ["HYP"]
}

def parse_scp_codes(scp_string):
    return ast.literal_eval(scp_string)

def scp_to_superclasses(scp_dict):
    labels = {k: 0 for k in SUPERCLASSES.keys()}
    for scp in scp_dict.keys():
        for supercls, scps in SUPERCLASSES.items():
            if scp.startswith(supercls):
                labels[supercls] = 1
    return labels

def encode_labels(df):
    y = []
    for scp_string in df["scp_codes"]:
        scp_dict = parse_scp_codes(scp_string)
        label_dict = scp_to_superclasses(scp_dict)
        y.append(list(label_dict.values()))
    return np.array(y)
