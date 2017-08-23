import sys, os
# 親ディレクトリのファイルをインポートする設定
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from sigmoid import sigmoid
from softmax import softmax
from identity import identity_function


# 手書き文字認識
# データセット：MNIST
# 0から9までの数字の画像から構成される

def get_data():
    # Return MNIST data.
    # (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    # normarize=Trueで正規化（0.0~1.0の間に値を収めてくれる）
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    # これ自分で書いてないわ．
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    # y = identity_function(a3)

    return y

if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    # バッチのサイズ
    batch_size = 100

    # 認識精度
    accuracy_cnt = 0
    # for i in range(start, end, step):
    for i in range(0, len(x), batch_size):
        # x[start_index:end_index]
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        # axis=1にすると1次元目を軸にして最大値を探す
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
