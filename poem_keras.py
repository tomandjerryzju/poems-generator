import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import CuDNNLSTM,Dense,Input,Softmax,Convolution1D,Embedding,Dropout,LSTM
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.models import Model
from utils import load,get_batch,predict_from_nothing,predict_from_head

UNITS = 256
batch_size = 64
epochs = 2
poetry_file = 'poetry.txt'

# 载入数据
x_data,char2id_dict,id2char_dict = load(poetry_file)
max_length = max([len(txt) for txt in x_data])
words_size = len(char2id_dict)

#-------------------------------#
#   建立神经网络
#-------------------------------#
inputs = Input(shape=(None,words_size))  # inputs shape: (batch_size, timesteps, input_dim)，这里的batch_size=None, timesteps=None，说明batch_size和timesteps都支持变长。会自动根据实际输入的shape进行推导。
# 当定义了LSTM输入的timesteps，输入仍然是按照顺序输入每个时间步的token到LSTM中，当然，这个过程是自动完成的。
# 但是，要注意，给LSTM输入一个时间步的输入，虽然它是RNN，但是不会自动无限循环下去，因为这是数字编程方式。因此，如果定义了LSTM输入的timesteps，那么，LSTM只会执行
# 前向计算timesteps次。如果timesteps=1，那么LSTM只会执行前向计算1次。
x = LSTM(UNITS,return_sequences=True)(inputs)
x = Dropout(0.6)(x)
x = LSTM(UNITS)(x)
x = Dropout(0.6)(x)
x = Dense(words_size, activation='softmax')(x)
model = Model(inputs,x)

#-------------------------------#
#   划分训练集验证集
#-------------------------------#
val_split = 0.1
np.random.seed(10101)
np.random.shuffle(x_data)
np.random.seed(None)
num_val = int(len(x_data)*val_split)
num_train = len(x_data) - num_val

#-------------------------------#
#   设置保存方案
#-------------------------------#
checkpoint = ModelCheckpoint('logs/loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)


#-------------------------------#
#   设置学习率并训练
#-------------------------------#
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])

"""
训练样本构造方式：见get_batch
    每个训练样本格式：(x, y)，其中，
        x shape (timesteps, input_dims), timesteps=6, input_dims=3001, input_dims即词汇表大小，这里采用的是one-hot形式。
        y shape (input_dims), input_dims=3001, input_dims即词汇表大小，这里采用的是one-hot形式。
    
    注意
        训练过程中，x全是真实数据，即不会利用模型的预测输出来构造x。但是预测的时候，会利用之前的模型预测输出作为输入，预测后面的结果。
        因此，训练阶段，lstm不会利用之前的模型预测输出作为输入，预测后面的结果。但是预测阶段会。
    
    这里
        每前6个字预测后一个字，因为一首诗不止7个字，通过滑动窗口方式(步长为1)，可以构造多个这样的训练样本。
        batch_size是指一个batch包含多少诗，因此，一个batch内，实际的训练样本数为batch_size首诗，每首诗能够组成的样本数总数。
"""
for i in range(epochs):
    predict_from_nothing(i,x_data,char2id_dict,id2char_dict,model)
    model.fit_generator(get_batch(batch_size, x_data[:num_train], char2id_dict, id2char_dict),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=get_batch(batch_size, x_data[num_train:], char2id_dict, id2char_dict),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=1,
                    initial_epoch=0,
                    callbacks=[checkpoint])


#-------------------------------#
#   设置学习率并训练
#-------------------------------#
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',
              metrics=['accuracy'])
        
for i in range(epochs):
    predict_from_nothing(i,x_data,char2id_dict,id2char_dict,model)
    model.fit_generator(get_batch(batch_size, x_data[:num_train], char2id_dict, id2char_dict),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=get_batch(batch_size, x_data[:num_train], char2id_dict, id2char_dict),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=1,
                    initial_epoch=0,
                    callbacks=[checkpoint])
