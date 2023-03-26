#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        #매개변수 설명
        #enc_unit: RNN의 뉴런 개수 (LSTM 레이어의 뉴런 개수)
        #          LSTM을 사용하기 때문에 h, c 
        #          즉 2*enc_unit 개의 파라미터를 가지게 된다.
        #batch_sz: 배치 크기 디폴트값
        #          최ㅇ초 Zero 상태 출력할 때 사용
        
        #부모 클래스(tf.keras.Model) 초기화 호출


        super(Encoder, self).__init__()
        #객체 변수 할당
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        
        #★★★★★★★★★★★★★★★★
        #CNN 추가 부분

	   #CNN 레이어 정의
        timeD = tf.keras.layers.TimeDistributed
        
        self.c1 = timeD(
          tf.keras.layers.Conv2D(enc_units//4,
                                 (3, 3),
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")
        )
        self.p1 = timeD(
          tf.keras.layers.MaxPooling2D((2, 2),
                                      strides=(2, 2))
        )
        
        self.c2 = timeD(
          tf.keras.layers.Conv2D(enc_units//2,
                                 (3, 3),
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")
        )
        self.p2 = timeD(
                tf.keras.layers.MaxPooling2D((2, 2),
                                            strides=(2, 2))
        )
        self.c3 = timeD(
          tf.keras.layers.Conv2D(enc_units,
                                 (3, 3),
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")
                       )
        self.p3 = timeD(
          tf.keras.layers.GlobalAveragePooling2D()
        )
        #★★★★★★★★★★★★★★★
        #RNN으로는 LSTM 사용 선언
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(self.enc_units, activation = 'tanh'),
            return_sequences = True, 
            return_state = True)
    
    #객체를 함수처럼 호출하면 실제로 실행되는 메소드
    # output, state = encoder(x, hidden)
    def call(self, x, hidden):
        #x :     [batch_size, time_step, n_features] 
        #x : 입력 데이터의 형태임
        #hidden: rnn cell에 대응되는 state 차원 // LSTM 셀의 초기 상태임
        # 인코더를 호출하는 함수임
        #output: [batch_size, times_step, enc_unit]
        # state는 마지막 time step의 Encoder의 상태
        # LSTM을 사용한다면, state는 [h, c] 두 벡터 원소를 
        # 가진 리스트로 출력된다
        #h:      [batch_size, enc_unit]
        #c:      [batch_size, enc_unit]
        # CNN 레이어를 통해 데이터를 인코딩함
        c = x
        c = self.c1(c)
        c = self.p1(c)
        c = self.c2(c)
        c = self.p2(c)
        c = self.c3(c)
        c = self.p3(c)
        
        # LSTM 레이어를 통해 시계열 데이터를 처리
        output, *state = self.rnn(c, initial_state = hidden)
        return output, state

    #인코더의 최초 상태(zero state)를 반환하는 메소드
    #LSTM을 사용하기 때문에, 원소 두개의 리스트를 반환함
    def initialize_hidden_state(self, batch_size = None):
        #return zero state
        #인코더의 초기 상태를 반환
        #LSTM을 사용하기 때문에, 원소 두 개의 리스트를 반환함 -> return 참조
        if (batch_size == None):
            return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]
        else:
            return [tf.zeros((batch_size, self.enc_units)), tf.zeros((batch_size, self.enc_units))]

#BahdanauAttention을 변형한 Pointer layer
class Pointer(tf.keras.layers.Layer):
    def __init__(self, units, return_as_logits = True):
        # unit: Dense 계층의 뉴런 개수
        # return_as_logits: 마지막 pointer의 출력을
        #           소프트맥스를 통과한 확률(False)로 할 것인가
        #           스코어 값(True)으로 할 것인가.
        super(Pointer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.return_as_logits = return_as_logits

    def call(self, query, keys):
        #query: (BATCH_SIZE, TIMESTEP, hidden_size) ((BATCH_SIZE, 1, hidden_size) at inference )
        #keys: (BATCH_SIZE, TABLESIZE, hidden_size), Table => End token + Encoder step
        
        time_step_length = query.shape[1]
        
         
        # enc_vec: (BATCH_SIZE, TABLESIZE, units)
        enc_vec = self.W1(keys)
        
        # enc_vec_with_time_axis: (BATCH_SIZE, TIMESTEP, TABLESIZE, units)
        enc_vec_with_time_axis = tf.tile(tf.expand_dims(enc_vec, 1), [1, time_step_length, 1, 1])
        
        #dec_vec: (BATCH_SIZE, TIMESTEP, units)
        dec_vec = self.W2(query)
        
        #dec_vec_for_broadcasting: (BATCH_SIZE, TIMESTEP, 1. units)
        dec_vec_for_broadcasting = tf.expand_dims(dec_vec, 2)
        
        
        #score: (BATCH_SIZE, TIMESTEP, TABLESIZE, 1)
        score = self.V(tf.nn.tanh(
            enc_vec_with_time_axis + dec_vec_for_broadcasting))
        
        
        if self.return_as_logits:
            return score
        
        # as Probablity
        # attention_weights: (BATCH_SIZE, TIMESTEP, TABLESIZE, 1)
        attention_weights = tf.nn.softmax(score, axis=2)

        return attention_weights

      
class Decoder(tf.keras.Model):
    def __init__(self, dec_units, batch_sz, return_as_logits = True):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        
        #★★★★★★★★★★★★★★
        timeD = tf.keras.layers.TimeDistributed
        
        self.c1 = timeD(
          tf.keras.layers.Conv2D(dec_units//4,
                                 (3, 3),
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")
        )
        self.p1 = timeD(
          tf.keras.layers.MaxPooling2D((2, 2),
                                      strides=(2, 2))
        )
        
        self.c2 = timeD(
          tf.keras.layers.Conv2D(dec_units//2,
                                 (3, 3),
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")
        )
        self.p2 = timeD(
                tf.keras.layers.MaxPooling2D((2, 2),
                                            strides=(2, 2))
        )
        self.c3 = timeD(
          tf.keras.layers.Conv2D(dec_units,
                                 (3, 3),
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")
                       )
        self.p3 = timeD(
          tf.keras.layers.GlobalAveragePooling2D()
        )
        
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(self.dec_units, activation = 'tanh'),
            return_sequences = True, 
            return_state = True)
        #★★★★★★★★★★★★★★
        # used for attention
        self.pointer = Pointer(self.dec_units, return_as_logits = return_as_logits)

    def call(self, x, hidden, values):
        #values: (batch_size, TABLESIZE, hidden_size)
        
        #output: (batch_size, dec_max_step, hidden_size )
        
        c = x
        c = self.c1(c)
        c = self.p1(c)
        c = self.c2(c)
        c = self.p2(c)
        c = self.c3(c)
        c = self.p3(c)
        
        output, *state = self.rnn(c, initial_state = hidden)
        
        #score: (BATCH_SIZE, , 1+ ENC_MAX_STEP, 1)
        score = self.pointer(output, values)

        return score, state

