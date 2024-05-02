import pandas as pd
import os

def get_gloss(num: int) -> str:
    """
    Return gloss given numeric value

    Parameters
    ----------
    num : int
        DESCRIPTION.

    Returns
    -------
    str

    """
        # df = pd.read_csv(os.path.join(BASE_DIR, 'translations', 'utils', 'ai_model','wlasl_class_list.txt'), sep='\t', header=None, names=["num", "word"])

    df = pd.read_csv('./preprocess/wlasl_class_list.txt', sep='\t', header=None, names=["num", "word"])
    return df[df['num'] == num]['word'].values[0]