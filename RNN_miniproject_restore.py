
# coding: utf-8

# In[9]:


'''
model을 restore하여 text를 generate하는 코드
실행 방법은 
1. model_output 폴더, RNN_ptb_data 폴더, RNN_miniproejct_restore.py를 HOME directory에 넣는다. 
2. cmd창에 python RNN_miniproject_restore.py --start_word=원하는 첫단어 --text_length=원하는 길이를 입력하면 된다.
   기본 설정은 start_word=it, text_length=200이다.
   (EX. python RNN_miniproject_restore.py --start_word=although --text_length=30)
   다만, start_word는 train dataset에 있는 단어들 중 하나로 입력해야 한다. 이들 중 하나일 경우 오류가 발생한다.

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import os
import sys

import inspect
import time
import random
import tensorflow as tf

Py3 = sys.version_info[0] == 3
data_path = 'RNN_ptb_data/'

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("start_word","it","the first word for making sentence")
flags.DEFINE_integer("text_length",200,"the text length you want to generate")

FLAGS = flags.FLAGS


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n","<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n","<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)
    
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1],x[0]))
    
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    
    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    #1. data_path로부터 PTB raw data를 Load
    #2. PTB file의 string을 integer ids로 바꾸고, mini-batching
    #PTB dataset from Tomas Mikolov's webpage: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    #Args: data_path 
    #Returns: tuple(train_data, valid_data, test_data, word_to_id, id_to_word, vocabulary)
    
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    word_to_id, id_to_word = _build_vocab(train_path)
    
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    
    vocabulary = len(word_to_id)
    
    return train_data, valid_data, test_data, word_to_id, id_to_word, vocabulary

def ptb_producer(raw_data, batch_size, num_steps, name=None):
    #Iterate on the raw PTB data
    #raw_data를 batches로 chunking하여 Tensor를 return
    #Args: raw_data - ptb_raw_data로부터 얻은 raw_data 중 1
    #      batch_size - int, batch size
    #      num_steps - int, the number of unrolls.
    #      name: 이 operation의 이름(optional)
    #Returns: A pair of Tensors - 각각 [batch_size,num_steps]
    #       2번째 원소: 같은 data를 time-shifted to the right by one
    #Raises: tf.erros.InvalidArgumentError - batch_size나 num_steps가 너무 클 때
    
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        
        #data를 batch-size에 맞게 reshape
        data = tf.reshape(raw_data[0:batch_size*batch_len],[batch_size, batch_len])
        
        epoch_size = (batch_len-1) //num_steps        
        #만약 epoch_size가 0보다 작거나 같을 때 에러발생(0보다 클 경우 그냥 None)
        #즉, batch_size나 num_steps가 너무 클 때 에러발생
        assertion = tf.assert_positive(epoch_size, message="epoch_size ==0, decrease batch_size or num_steps")        
        with tf.control_dependencies([assertion]):
            epoch_size=tf.identity(epoch_size,name="epoch_size")
        
        #0부터 epoch_size-1까지의 queue생성 후 dequeue    
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        
        #data를 num_steps로 slice
        x = tf.strided_slice(data, [0,i*num_steps], [batch_size, (i+1)*num_steps])
        x.set_shape([batch_size,num_steps])
        
        #오른쪽으로 한칸 옮겨서 slice 
        y = tf.strided_slice(data, [0,i*num_steps+1],[batch_size, (i+1)*num_steps+1])
        y.set_shape([batch_size,num_steps])
        
        return x,y
    
#ptb data Load
raw_data = ptb_raw_data(data_path)
train_data, valid_data, test_data, word_to_id, id_to_word, vocab_size = raw_data


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class SmallConfig(object):
    #Small Config
    init_scale=0.1 #the initial scale of the weights
    learning_rate=1.0 #the initial value of the learning rate
    max_grad_norm=5 #max gradient
    num_layers=2  #number of LSTM layers
    num_steps=20 #sequence length; the number of unrolls
    hidden_size=200 #number of hidden units in LSTM; also embedding size
    max_epoch=4 #the number of epochs trained with the initial learning rate
    max_max_epoch=13 #the total number of epochs for training
    keep_prob=1.0 #1-dropoff rate
    lr_decay=0.5 #the decay of the learning rate for each epoch after "max_epoch"
    batch_size=20
    vocab_size=10000
    
    
class MediumConfig(object):
    #Medium Config
    init_scale=0.05
    learning_rate=1.0
    max_grad_norm=5
    num_layers=2
    num_steps=35
    hidden_size=650
    max_epoch=6
    max_max_epoch=39
    keep_prob=0.5
    lr_decay=0.8
    batch_size=20
    vocab_size=10000
    
    
class LargeConfig(object):
    #Large Config 
    init_scale=0.04
    learning_rate=1.0
    max_grad_norm=10
    num_layers=2
    num_steps=35
    hidden_size=1500
    max_epoch=14
    max_max_epoch=55
    keep_prob=0.35
    lr_decay=1/1.15
    batch_size=20
    vocab_size=10000
    
    
class TestConfig(object):
    #Tiny Config for testing: 제대로 작동하는지를 확인하기 위한 빠른 코드
    init_scale=0.1
    learning_rate=1.0
    max_grad_norm=1
    num_layers=1
    num_steps=2
    hidden_size=2
    max_epoch=1
    max_max_epoch=1
    keep_prob=1.0
    lr_decay=0.5
    batch_size=20
    vocab_size=10000
    
    
class PTBInput(object):
    #The input data
    
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size)-1) // num_steps
        self.input_data, self.targets = ptb_producer(data, batch_size, num_steps, name=name)     

class PTBModel(object):
    #The PTB model

    def __init__(self, is_training, config, input_=None):
        self._is_training = is_training
        batch_size = config.batch_size
        num_steps = config.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        
        if input_ is not None:
            # For normal training and validation
            self._input = input_
            self._input_data = input_.input_data
            self._targets = input_.targets
            
        else:
            # For text generations
            self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
            self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        #LSTMCell hiden_size 만큼 forget bias =0으로 생성
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                hidden_size,
                forget_bias=0.0,
                state_is_tuple=True,
                reuse = not is_training)

            # Note because we set `state_is_tuple=True`, the states are 2-tuples of the `c_state` and `h_state`
            # `c_state` is the cell state
            # `h_state` is the hidden state
            # See this SO thread: https://stackoverflow.com/questions/41789133/c-state-and-m-state-in-tensorflow-lstm
    
        #기본적으론 lsm_cell
        attn_cell = lstm_cell
        
        # Implement dropoff (for training only)
        if is_training and config.keep_prob < 1:

            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        # Stacking multiple LSTMs
        attn_cells = [attn_cell() for _ in range(config.num_layers)]
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(attn_cells, state_is_tuple=True)
        
        # Initialize states with zeros
        # `_initial_state` is a list of `num_layers` tensors
        # Each is a tuple of (`c_state`, `h_state`),
        # and both `c_state` and `h_state` are shaped [batch_size, hidden_size]
        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
        
        # The word IDs will be embedded into a dense representation before feeding to the LSTM.
        # This allows the model to efficiently represent the knowledge about particular words.
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, hidden_size], dtype=tf.float32)
            input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
            # The shape of `input_embeddings` is [batch_size, num_steps, hidden_size]
        
        # Implement dropoff (for training only)
        if is_training and config.keep_prob < 1:
            input_embeddings = tf.nn.dropout(input_embeddings, config.keep_prob)
        
        # Unroll LSTM loop
        outputs = []
        state = self._initial_state
        
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                
                (cell_output, state) = stacked_lstm(input_embeddings[:, time_step, :], state)
                outputs.append(cell_output)
        # `outputs` is a list of `num_steps` tensors, each shaped [batch_size, hidden_size]
        # Resize the ouput into a [batch_size * num_steps, hidden_size] matrix.
        # Note axis=1 because we want to group words together according to its original sequence
        # in order to compare with `targets` to compute loss later.
        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_size])
        
        # Compute logits
        softmax_w = tf.get_variable(
            "softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=tf.float32)
        
        self._logits = logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # The shape of `logits` =
        # [batch_size * num_steps, hidden_size] x [hidden_size, vocab_size] + [vocab_size] =
        # [batch_size * num_steps, vocab_size]

        # Sample based on the size of logits (used for text generation)
        self._logits_sample = tf.multinomial(logits, 1)
     
        # Reshape logits to be 3-D tensor for sequence loss
        logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        # loss함수
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,  # shape: [batch_size, num_steps, vocab_size]
            self._targets,  # shape: [batch_size, num_steps]
            tf.ones([batch_size, num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost variables and state 
        self._cost = cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        # Optimizer
        self._lr = tf.Variable(0.0, trainable=False)
        
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars), config.max_grad_norm)
        
        #GradientDescent방법으로 Optimize. learning rate = self._lr
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        
        #new learning rate를 위한 placeholder
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        #self._lr 에 new learning rate assign 
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
        
    # Learning rate update
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    
    
    # properties
    @property
    def input(self):
        return self._input
    
    @property
    def input_data(self):
        return self._input_data
    
    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
    
    @property
    def logits_sample(self):
        return self._logits_sample
    
    @property
    def logits_max(self):
        return self._logits_max
    
    @property
    def logits(self):
        return self._logits

#Text generations 
#by sampling
def generate_text_by_sampling(session, model, feed, text_length):
    state = session.run(model.initial_state)
    fetches = {
        "final_state": model.final_state,
        "logits": model.logits_sample
    } 
    
    generated_text = [feed]
    
    for i in range(text_length):
        feed_dict = {}
        feed_dict[model.input_data] = feed
        
        for i, (c,h) in enumerate(model.initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
            
        vals = session.run(fetches, feed_dict)
        
        #Extract final_state and sampled logits after the current step,
        #which become the new state and feed for the next step
        state = vals["final_state"]
        feed = vals["logits"]
        
        #Append generated text
        generated_text.append(feed)
        
    return generated_text


# In[10]:


#본래의 코드에서 이 main만 바꿔서 실행

def main(_):
    
    #입력한 start_word와 text_length 저장
    start_word = FLAGS.start_word
    start_word = np.array(word_to_id[start_word]).reshape(1,1)
    text_length = FLAGS.text_length

    #현재 Model이 small이기에
    config = SmallConfig()
    #Text generation을 위한 config
    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    
    #model이 저장된 폴더 경로
    model_path = 'model_output'

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    
    # Define model for text generations
        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m=PTBModel(is_training=True, config=config, input_=train_input)            
        
        with tf.name_scope("Feed"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mfeed = PTBModel(is_training=False, config=eval_config)

        sv = tf.train.Supervisor(logdir=model_path)
        with sv.managed_session() as session:
            # Restore model weights from previously saved model
            ckpt = tf.train.get_checkpoint_state(model_path)
            sv.saver.restore(session, ckpt.model_checkpoint_path)
            print("Model restored from file: %s\n" % ckpt.model_checkpoint_path)
        
            generated_text = generate_text_by_sampling(session, mfeed, start_word, text_length)
            generated_text = ' '.join([id_to_word[text[0, 0]] for text in generated_text])
            print ("Sample text generation:\n", generated_text)
            
if __name__ =="__main__":
    tf.app.run()


# In[5]:




