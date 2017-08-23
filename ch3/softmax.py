import numpy as np
# 分類問題（データの属するクラスはどれか）の出力層で用いられる
# ソフトマックス関数の出力は0から1.0の間の実数を取る
# また，出力の総和は1になる．（出力は確率としても見ることができる）
def softmax(a):
    c = np.max(a)
    # To take measures against an overflow.
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
