from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import Masking, Embedding
from keras.layers.wrappers import Bidirectional,TimeDistributed
from functools import reduce
import re

from attention_utils import get_activations, get_data_recurrent



# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longerZ
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def vectorize_stories(input_list, tar_list, word_idx, input_maxlen, tar_maxlen, vocab_size):
    x_set = []
    Y = np.zeros((len(tar_list), tar_maxlen, vocab_size), dtype=np.bool)
    for _sent in input_list:
        x = [word_idx[w] for w in _sent]
        x_set.append(x)
    for s_index, tar_tmp in enumerate(tar_list):
        for t_index, token in enumerate(tar_tmp):
            Y[s_index, t_index, word_idx[token]] = 1

    return pad_sequences(x_set, maxlen=input_maxlen), Y
    #return x_set, Y
def vectorize_stories1(input_list, tar_list, word_idx, input_maxlen, tar_maxlen, idx_to_word):
    x_set = []
    y_set = []
    #Y = np.zeros((len(tar_list), tar_maxlen, vocab_size), dtype=np.bool)
    for _sent in input_list:
        x = [word_idx[w] for w in _sent]
        x_set.append(x)
    for _sent in input_list:
        y = [word_idx[w] for w in _sent]
        y_set.append(y)
#    for s_index, tar_tmp in enumerate(tar_list):
#        for t_index, token in enumerate(tar_tmp):
#            Y[s_index, t_index, word_idx[token]] = 1

    return pad_sequences(x_set, maxlen=input_maxlen), pad_sequences(y_set, maxlen=tar_maxlen)

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    #input_dim = int(inputs.shape[2])tar_maxlen
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(input_maxlen,))
    embed = Embedding(input_dim=vocab_size,
                              output_dim=620,
                              input_length=input_maxlen)(inputs)
    attention_mul = LSTM(hidden_dim, return_sequences=True)(embed)    
    attention_mul = attention_3d_block(attention_mul)
    attention_mul = LSTM(500, return_sequences=True)(attention_mul)
    output = TimeDistributed(Dense(output_dim, activation='sigmoid'),input_shape=(tar_maxlen, output_dim))(attention_mul)
    #output = TimeDistributed(Dense(output_dim))(output)
    #output = Dense(output_dim, activation='sigmoid')(attention_mul)
    #output = RepeatVector(tar_maxlen)(output)
    #output = Permute((tar_maxlen,output_dim),name='reshapeLayer')(output)
    output = Activation('softmax')(output)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':
    input_text = ['中国 的 首都 是 北京'
                  , '日本 的 首都 是 东京'
                  , '美国 的 首都 是 华盛顿'
                  , '英国 的 首都 是 伦敦'
                  , '德国 的 首都 是 柏林']
    
    tar_text = ['Beijing is the capital of China'
                , 'Tokyo is the capital of Japan'
                , 'Washington is the capital of the United States'
                , 'London is the capital of England'
                , 'Berlin is the capital of Germany']
                
    input_list = []
    tar_list = []
    END = ' EOS'
    for tmp_input in input_text:
        tmp_input = tmp_input+END
        input_list.append(tokenize(tmp_input))
    for tmp_tar in tar_text:
        tmp_tar = tmp_tar+END
        tar_list.append(tokenize(tmp_tar))
    

    vocab = sorted(reduce(lambda x, y: x | y, (set(tmp_list) for tmp_list in input_list + tar_list)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1  # keras进行embedding的时候必须进行len(vocab)+1
    input_maxlen = max(map(len, (x for x in input_list)))
    tar_maxlen = max(map(len, (x for x in tar_list)))
    output_dim = vocab_size
    hidden_dim = 1000
    INPUT_DIM = hidden_dim #senten length
    TIME_STEPS = input_maxlen #
# If I have 1000 sentences ,each sentence has 10 words, and each word is presented in a 3-dim vector, 
#so 1000 is nb_samples, 10 is the timesteps and 3 is the input_dim
    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Input max length:', input_maxlen, 'words')
    print('Target max length:', tar_maxlen, 'words')
    print('Dimension of hidden vectors:', hidden_dim)
    print('Number of training stories:', len(input_list))
    print('Number of test stories:', len(input_list))
    print('-')
    print('Vectorizing the word sequences...')
    word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))  # 编码时需要将字符映射成数字index
    idx_to_word = dict((i + 1, c) for i, c in enumerate(vocab))  # 解码时需要将数字index映射成字符
    inputs_train, tars_train = vectorize_stories(input_list, tar_list, word_to_idx, input_maxlen, tar_maxlen, vocab_size)
#    N = 300000  #the number of sample 
    # N = 300 -> too few = no training
#    inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)

#    if APPLY_ATTENTION_BEFORE_LSTM:
#        m = model_attention_applied_after_lstm()
#    else:
#        m = model_attention_applied_before_lstm()
    m = model_attention_applied_before_lstm()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit(inputs_train, tars_train, epochs=10, batch_size=3, validation_split=0.1)
#fit(inputs_train, tars_train, batch_size=3, nb_epoch=1, show_accuracy=True)
#    attention_vectors = []
#    for i in range(300):
#        testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
#        attention_vector = np.mean(get_activations(m,
#                                                   testing_inputs_1,
#                                                   print_shape_only=True,
#                                                   layer_name='attention_vec')[0], axis=2).squeeze()
#        print('attention =', attention_vector)
#        assert (np.sum(attention_vector) - 1.0) < 1e-5
#        attention_vectors.append(attention_vector)
#
#    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
#    # plot part.
#    import matplotlib.pyplot as plt
#    import pandas as pd
#
#    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
#                                                                         title='Attention Mechanism as '
#                                                                               'a function of input'
#                                                                               ' dimensions.')
#    plt.show()
