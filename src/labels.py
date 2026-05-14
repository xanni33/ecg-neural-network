#scp + 5 klas
# src/labels.py
import ast
import numpy as np

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def parse_scp_codes(scp_string):
    return ast.literal_eval(scp_string)


def encode_labels(df, statements_df):
    """
    Tworzy multi-label vector dla PTB-XL.
    """

    y = []

    for scp_string in df["scp_codes"]:

        scp_dict = parse_scp_codes(scp_string)

        labels = [0] * len(SUPERCLASSES)

        for scp_code in scp_dict.keys():

            if scp_code in statements_df.index:

                diagnostic_class = statements_df.loc[scp_code].get(
                    "diagnostic_class",
                    None
                )

                if diagnostic_class in SUPERCLASSES:
                    idx = SUPERCLASSES.index(diagnostic_class)
                    labels[idx] = 1

        y.append(labels)

    return np.array(y, dtype=np.float32)