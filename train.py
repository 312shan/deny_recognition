#! -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
import jieba
import pandas as pd
import gc

mode = 0
min_count = 2
char_size = 128
num_features = 3

df = pd.read_csv("data/train.tsv", sep="\t")
print(df.columns)
# df = df.head(1000) # 调试用
# 划分训练集和开发集
L = len(df)
df = df.sample(n=L)
train_size = int(0.9 * L)
train_df = df.head(train_size)
dev_df = df.tail(L - train_size)

# 构建字符和id映射
texts = list(df["Question"]) + list(df["Description"]) + list(df["Answer"])
del df

gc.collect()
char_set = set()
for t in texts:
    char_set.update(t)
print(len(char_set))
id2char = {i + 2: j for i, j in enumerate(char_set)}
char2id = {j: i for i, j in id2char.items()}

# 加载词向量和映射
word2vec = Word2Vec.load('D:\\nlp\\data\\kg\\word2vec_baike\\word2vec_baike')
id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}
word2vec = word2vec.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])


def tokenize(s):
    """如果pyhanlp不好用，自己修改tokenize函数，
    换成自己的分词工具即可。
    """
    return jieba.lcut(s)  # [i.word for i in pyhanlp.HanLP.segment(s)]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
        for x in X
    ])


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[(-1)].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, X1V, X2V, T = (
                [], [], [], [], []
            )
            for i in idxs:
                d = self.data.iloc[i, :]
                question = d['Question'].lower() + '[SEP]' + d["Description"].lower()
                answer = d['Answer'].lower()
                lb = int(d["Label"])

                question_words = tokenize(question)
                question = ''.join(question_words)
                x1 = [char2id.get(c, 1) for c in question]  # query.char_ids

                answer_words = tokenize(answer)
                answer = ''.join(answer_words)
                x2 = [char2id.get(c, 1) for c in answer]  # query.char_ids

                X1.append(x1)  # query.char_ids
                X2.append(x2)  # document.char_ids
                X1V.append(question_words)  # query.words
                X2V.append(answer_words)  # document.words
                T.append([lb])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    X1V = sent2vec(X1V)
                    X2V = sent2vec(X2V)
                    T = seq_padding(T)
                    yield [X1, X2, X1V, X2V, T], None
                    X1, X2, X1V, X2V, T = (
                        [], [], [], [], []
                    )


class dev_data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        X1, X2, X1V, X2V, T = (
            [], [], [], [], []
        )
        for i in idxs:
            d = self.data.iloc[i, :]
            question = d['Question'].lower() + '[SEP]' + d["Description"].lower()
            answer = d['Answer'].lower()
            lb = int(d["Label"])

            question_words = tokenize(question)
            question = ''.join(question_words)
            x1 = [char2id.get(c, 1) for c in question]  # query.char_ids

            answer_words = tokenize(answer)
            answer = ''.join(answer_words)
            x2 = [char2id.get(c, 1) for c in answer]  # query.char_ids

            X1.append(x1)  # query.char_ids
            X2.append(x2)  # document.char_ids
            X1V.append(question_words)  # query.words
            X2V.append(answer_words)  # document.words
            T.append([lb])
            if len(X1) == self.batch_size or i == idxs[-1]:
                X1 = seq_padding(X1)
                X2 = seq_padding(X2)
                X1V = sent2vec(X1V)
                X2V = sent2vec(X2V)
                T = seq_padding(T)
                yield [X1, X2, X1V, X2V, T], None
                X1, X2, X1V, X2V, T = (
                    [], [], [], [], []
                )


class test_data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        X1, X2, X1V, X2V, DOC_ID = (
            [], [], [], [], []
        )
        for i in idxs:
            d = self.data.iloc[i, :]
            question = d['Question'].lower() + '[SEP]' + d["Description"].lower()
            answer = d['Answer'].lower()
            doc_id = int(d["Docid"])

            question_words = tokenize(question)
            question = ''.join(question_words)
            x1 = [char2id.get(c, 1) for c in question]  # query.char_ids

            answer_words = tokenize(answer)
            answer = ''.join(answer_words)
            x2 = [char2id.get(c, 1) for c in answer]  # query.char_ids

            X1.append(x1)  # query.char_ids
            X2.append(x2)  # document.char_ids
            X1V.append(question_words)  # query.words
            X2V.append(answer_words)  # document.words
            DOC_ID.append(doc_id)
            if len(X1) == self.batch_size or i == idxs[-1]:
                X1 = seq_padding(X1)
                X2 = seq_padding(X2)
                X1V = sent2vec(X1V)
                X2V = sent2vec(X2V)
                yield [X1, X2, X1V, X2V, DOC_ID], None
                X1, X2, X1V, X2V, DOC_ID = (
                    [], [], [], [], []
                )


train_D = data_generator(train_df)
dev_D = dev_data_generator(dev_df)
for row, _ in train_D:
    for r in row:
        print(r.shape)
    break
"""
(64, 19)
(64, 303)
(64, 19, 256)
(64, 303, 256)
(64, 1)
"""

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

K.clear_session()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)


# https://kexue.fm/archives/4765
class Attention(Layer):
    """多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(
            name='q_kernel',
            shape=(q_in_dim, self.out_dim),
            initializer='glorot_normal')
        self.k_kernel = self.add_weight(
            name='k_kernel',
            shape=(k_in_dim, self.out_dim),
            initializer='glorot_normal')
        self.v_kernel = self.add_weight(
            name='w_kernel',
            shape=(v_in_dim, self.out_dim),
            initializer='glorot_normal')

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = (None, None)
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class MyBidirectional:
    """自己封装双向RNN，允许传入mask，保证对齐
    """

    def __init__(self, layer):
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, inputs):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        x, mask = inputs
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)

    def __call__(self, inputs):
        x, mask = inputs
        x_forward = self.forward_layer(x)
        x_backward = Lambda(self.reverse_sequence)([x, mask])
        x_backward = self.backward_layer(x_backward)
        x_backward = Lambda(self.reverse_sequence)([x_backward, mask])
        x = Concatenate()([x_forward, x_backward])
        x = Lambda(lambda x: x[0] * x[1])([x, mask])
        return x


x1_in = Input(shape=(None,), name='x1_in')
x2_in = Input(shape=(None,), name='x2_in')
x1v_in = Input(shape=(None, word_size), name='x1v_in')
x2v_in = Input(shape=(None, word_size), name='x2v_in')
t_in = Input(shape=(1,), name='t_in')

x1, x2, x1v, x2v, t = (
    x1_in, x2_in, x1v_in, x2v_in, t_in
)

# [B,CT]=>[B,CT,1]
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x1_mask')(x1)
# [B,WT]=>[B,WT,1]
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x2_mask')(x2)

embedding = Embedding(len(id2char) + 2, char_size, name='char_emb')
dense = Dense(char_size, use_bias=False, name='char_dense')

# 问题编码
x1 = embedding(x1)  # [B,CT]=>[B,CT,C_H]
x1v = dense(x1v)  # [B,WT,W_H]=>[B,WT,C_H] , 这里有一些预处理上的做法，导致这里 WT == CT 所以下面才能 Add
x1 = Add()([x1, x1v])  # query 的 char_level + word_level
x1 = Dropout(0.2)(x1)
x1 = Lambda(lambda x: x[0] * x[1], name='x1')([x1, x1_mask])
x1 = MyBidirectional(LSTM(char_size // 2, return_sequences=True, name='lstm_1'))([x1, x1_mask])

# context 编码
x2 = embedding(x2)
x2v = dense(x2v)
x2 = Add()([x2, x2v])
x2 = Dropout(0.2)(x2)
x2 = Lambda(lambda x: x[0] * x[1], name='x2')([x2, x2_mask])
x2 = MyBidirectional(LSTM(char_size // 2, return_sequences=True, name='lstm3'))([x2, x2_mask])

# attention attain to each other
x12 = Attention(8, 16, name='x12_att')([x1, x2, x2, x2_mask, x1_mask])
x12 = Lambda(seq_maxpool, name='x12')([x12, x1_mask])
x21 = Attention(8, 16, name='x21_att')([x2, x1, x1, x1_mask, x2_mask])
x21 = Lambda(seq_maxpool, name='x21')([x21, x2_mask])
x = Concatenate()([x12, x21])
x = Dropout(0.2)(x)
x = Dense(char_size, activation='relu', name='last')(x)
pt = Dense(1, activation='sigmoid')(x)

t_model = Model([x1_in, x2_in, x1v_in, x2v_in], pt)

train_model = Model(
    [x1_in, x2_in, x1v_in, x2v_in, t], [pt]
)
bce_loss = K.mean(K.binary_crossentropy(t, pt))

train_model.add_loss(bce_loss)
train_model.compile(optimizer=Adam(3e-3))
train_model.summary()


# https://kexue.fm/archives/6575#%E6%9D%83%E9%87%8D%E6%BB%91%E5%8A%A8%E5%B9%B3%E5%9D%87
class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """

    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(list(zip(self.ema_weights, self.old_weights)))

    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(list(zip(self.model.weights, ema_weights)))

    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(list(zip(self.model.weights, self.old_weights)))


EMAer = ExponentialMovingAverage(train_model)
EMAer.inject()

from sklearn.metrics import classification_report


# https://kexue.fm/archives/5765#%E8%8A%B1%E5%BC%8F%E5%9B%9E%E8%B0%83%E5%99%A8
class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            t_model.save_weights('best_model.weights')
        print(('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best)))
        EMAer.reset_old_weights()

    def evaluate(self):
        p, r, f1 = [], [], []
        pbar = tqdm()
        for d in dev_D:
            y_pred_p = t_model.predict(d[0][:-1])
            y_pred = np.asarray(y_pred_p > 0.5, dtype=int)
            y_true = d[0][-1]
            report = classification_report(y_true, y_pred, output_dict=True)["weighted avg"]
            p_, r_, f1_ = report["precision"], report["recall"], report["f1-score"]
            p.append(p_)
            r.append(r_)
            f1.append(f1_)
            pbar.update(y_true.shape[0])
            pbar.set_description('< f1: %.4f, precision: %.4f, recall: %.4f >' % (f1_, p_, r_))
        pbar.close()

        return np.mean(p), np.mean(r), np.mean(f1)


evaluator = Evaluate()
md_path = 'best_model.weights'
if __name__ == '__main__':

    if os.path.exists(md_path):
        train_model.load_weights(md_path)
    train_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=120,
        callbacks=[evaluator]
    )
else:
    train_model.load_weights(md_path)
    test_df = pd.read_csv("data/test.tsv", sep="\t")
    test_D = test_data_generator(test_df)
    doc_ids = []
    labels = []
    for d in tqdm(test_D):
        y_pred_p = t_model.predict(d[0][:-1])
        y_pred = np.asarray(y_pred_p > 0.5, dtype=int)
        y_pred = y_pred.flatten()
        labels.extend(y_pred)
        doc_ids.extend(d[0][-1])
    lb_df = pd.DataFrame({"Docid": doc_ids, "Label": labels})

    test_final = pd.merge(test_df, lb_df, on='Docid')
    cols = ['Label', 'Docid', 'Question', 'Description', 'Answer']
    test_final[cols].to_csv('cys_valid_result.txt', encoding="utf-8", sep='\t', index=False)
