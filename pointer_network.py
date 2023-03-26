#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import tensorflow as tf
import attention_models

#Define Token
START_ID = 0 #시퀀스 시작 토큰 ID
PAD_ID = 1   #시퀀스 공백 토큰 ID
END_ID = 2   #시퀀스 종결 토큰 ID

#Parameter
#입력 최대 길이
ENC_MAX_STEP = 10
#Label 최대 길이
DATA_MAX_OUTPUT = 4

DATA_PATH = "./data/convex_hull_5_test.txt"
PREFIX = "thermo"
#Label + End Token 최대 길이 
DEC_MAX_STEP = DATA_MAX_OUTPUT + 1 #max output lengths

BATCH_SIZE = 32
EPOCHS = 100
UNITS = 256
LEARNING_RATE = 0.001
BEAM_WIDTH = 4
OUTPUT_STD = None
DROPOUT_RATE = 0.0
#graident Clipping 파라미터
CLIPPING_VALUE = None
TRAIN_DATA_RATIO = 0.9

class PointerNetwork:
    def __init__(self, unit, batch_size,
                 learning_rate = 0.001,
                 output_std = None,
                 dropout_rate = 0.0):
        self.encoder = attention_models.Encoder(unit, batch_size)
        self.decoder = attention_models.Decoder(unit, batch_size, return_as_logits = True)
        #Loss Function에 사용할 함수로 Cross Entropy 사용
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(
                                      from_logits=True, reduction='none') 
        #학습 Optimizer 설정. Adam 사용
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        #End token의 Key token 생성
        #make end token embedding for pointing
        if (output_std is None):
            # 0.0185 == output's std when unit = 256
            output_std = 0.0185 * np.sqrt(256 / unit)
        self.end_token_outputs = tf.Variable(np.random.normal(0, output_std, [1, 1, unit]), dtype = np.float32)
        
        #token 좌표 테이블
        #디코더의 매 스텝마다 출력이 나오면, 해당 출력에 해당하는 점의 좌표 값이 
        #다음 스텝으로 입력으로 들어가게 되는데
        #시작 토큰(0)이나 다른 토큰(1, 2)들이 나왔을 때, 다음 입력으로 들어갈 (x, y) 좌표 테이블이다.
        #임의로 설정 가능하며, (0 , 0)으로 설정함
        #token_table = [[begin_token, end_token, pad_token]]
        #begin_token = end_token = pad_token = (0, 0)
        #★★★★★
        #원래 token에 대응되는 좌표 값들이였는데, token에 대한 이미지 값으로 차원 맞춰줌
        self.token_table = tf.Variable(np.zeros([1, 3, 40, 50, 1]), dtype = np.float32)
        
        self.dropout_rate = dropout_rate
        
        #Checkpoint 파라미터 전달용 dict
        self.model = {}
        self.model['encoder'] = self.encoder
        self.model['decoder'] = self.decoder
        self.model['optimizer'] = self.optimizer
        self.model['end_token_outputs'] = self.end_token_outputs
        
    def get_model(self):
        return self.model
    
    #정답 real(정수값)와 스코어 pred(실수 점수)가 들어오면 두 값의 loss를 구함 
    def loss_function(self, real, pred):
        #real.shape = (BATCH_SIZE, DEC_MAX_STEP)
        #pred.shape = (BATCH_SIZE, DEC_MAX_STEP, 1 + ENC_MAX_STEP, 1)
        
        #real_onehot.shape = (BATCH_SIZE, DEC_MAX_STEP, 1 + ENC_MAX_STEP)
        real_onehot = tf.one_hot(real, pred.shape[-2], on_value = 1.0)
        
        #loss_ = (BATCH_SIZE, DEC_MAX_STEP)
        loss_ = self.loss_object(real_onehot, pred)
        
        #real(정답)에서 padding(-1)에 해당하는 부분은 제외함
        #mask = (BATCH_SIZE, DEC_MAX_STEP)
        mask = tf.math.logical_not(tf.math.equal(real, -1))
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask) , loss_, mask
    
    #Teacher forcing
    #학습의 용이를 위하여 이전 스텝의 정답(targ[t-1])을 이번 스텝의 입력으로 넣는 학습법
    def teacher_forcing_run(self, inp, targ):
        #inp:  [BATCH_SIZE, ENC_MAX_STEP, 2]
        #targ: [BATCH_SIZE, DEC_MAX_STEP]
      
        #local batch size
        local_bsz = inp.shape[0]
        
        #zero state
        enc_state = self.encoder.initialize_hidden_state(local_bsz) 
        
        #Encoder 출력 파트 
        #enc_output.shape = (BATCH_SIZE, ENC_MAX_STEP, hidden_size)
        enc_output, enc_state = self.encoder(inp, enc_state)
        
        #Dropout
        if self.dropout_rate != 0.0:
            enc_output = tf.nn.dropout(enc_output, rate = self.dropout_rate)
            #LSTM 대응
            if type(enc_state) == list:
                for i in range(len(enc_state)):
                    enc_state[i] = tf.nn.dropout(enc_state[i], rate = self.dropout_rate)
            #LSTM이 아닌 GLU같은거
            else:
                enc_state = tf.nn.dropout(enc_state, rate = self.dropout_rate )
        
        #enc_output with end token output
        # .shape = (BATCH_SIZE, 1 + ENC_MAX_STEP, hidden_size )
        enc_output = tf.concat([tf.tile(self.end_token_outputs, [local_bsz, 1, 1]),
                            enc_output], 1)
        
        # token table에 대한 자세한 설명은 __init__ 함수 참고
        #tiled_token_table = (BATCH_SIZE, 3, 2)
        tiled_token_table = tf.tile(self.token_table, [local_bsz, 1, 1, 1, 1])
        #enc_table = (BATCH_SIZE, 3 + ENC_MAX_STEP, 2)
        #[token_table | input_table]
        enc_table = tf.concat([tiled_token_table, inp], 1)

        #시작 입력 (시작 토큰)
        #dec_input = (BATCH_SIZE, 1)
        dec_input = tf.expand_dims([START_ID] * local_bsz, 1)
        
        #teacher forcing 입력 설계
        #매 스텝의 입력은 다음과 같이 처리된다.
        #[start_token, first_label, ... last_label, pad_token, ...]
        
        #dec_input = (BATCH_SIZE, DEC_MAX_STEP), 정수형
        dec_input = tf.concat([dec_input, tf.cast(targ[:, :-1] + END_ID, dtype=np.int32)], 1)
        #dec_input = (BATCH_SIZE, DEC_MAX_STEP, 1)
        dec_input = tf.expand_dims(dec_input,-1)
        
        #dec_input = (BATCH_SIZE, DEC_MAX_STEP, 2)
        #좌표 테이블로 부터 id에 대응되는 (x, y) 좌표를 얻어낸다.
        dec_input = tf.gather_nd(enc_table, dec_input, batch_dims=1)
        #predictions = (BATCH_SIZE, DEC_MAX_STEP, 1 + ENC_MAX_STEP, 1)
        #encoder => enc_state => decoder
        #                         ^
        #                      x ┚
        predictions, dec_state = self.decoder(dec_input, enc_state, enc_output)
        
        total_loss, losses, masks = self.loss_function(targ, predictions) 
        return predictions, total_loss, losses, masks
    
    #without teacher forcing
    def naive_run(self, inp, targ,
                  return_as_idx = False,
                  max_length = 99999,
                  FORCED_TO_MAKE_TRIANGLE = False,
                  **kwargs): 
        #local batch size
        local_bsz = inp.shape[0]
        #zero state
        enc_state = self.encoder.initialize_hidden_state(local_bsz) 
        #enc_output.shape = (BATCH_SIZE, ENC_MAX_STEP, hidden_size)
        enc_output, enc_state = self.encoder(inp, enc_state)
        #enc_output with end token output
        # .shape = (BATCH_SIZE, 1 + ENC_MAX_STEP, hidden_size )
        enc_output = tf.concat([tf.tile(self.end_token_outputs, [local_bsz, 1, 1]),
                            enc_output], 1)
        
        # token table에 대한 자세한 설명은 __init__ 함수 참고
        #tiled_token_table = (BATCH_SIZE, 3, 2)
        tiled_token_table = tf.tile(self.token_table, [local_bsz, 1, 1, 1, 1])
        #enc_table = (BATCH_SIZE, 3 + ENC_MAX_STEP, 2)
        #[token_table | input_table]
        enc_table = tf.concat([tiled_token_table, inp], 1)

        #dec_input = (BATCH_SIZE, 1)
        dec_input = tf.expand_dims([START_ID] * local_bsz, 1)
        #encoder => state = > decoder
        dec_state = enc_state
        predictions = []
        predictions_idx = []
        losses = []
        total_loss = 0
        masks = []
        max_timesteps = max_length if targ is None else min(targ.shape[1], max_length)
        
        for t in range(0, max_timesteps):
            #id - > (x, y)
            #dec_input = (BATCH_SIZE, 2)
            dec_input = tf.gather_nd(enc_table, dec_input, batch_dims=1)
            #dec_input = (BATCH_SIZE, time_axis = 1, 2)
            dec_input = tf.expand_dims(dec_input, 1)
            #predictions = (BATCH_SIZE, 1, 1 + ENC_MAX_STEP, 1)
            prediction, dec_state = self.decoder(dec_input, dec_state, enc_output)
            
            #loss = (BATCH_SIZE, 1)
            #mask = (BATCH_SIZE, 1)
            loss_sum, loss, mask = self.loss_function(targ[:, t:t+1], prediction)
            total_loss += loss_sum
            losses.append(loss)
            masks.append(mask)
            #predicted_idx.shape = (BATCH_SIZE, 1, 1)
            predicted_idx = tf.argmax(prediction, axis = 2)
            dec_input = predicted_idx[:, :, 0] + END_ID
            predictions_idx.append(predicted_idx)
            predictions.append(prediction)
         
        predictions = tf.concat(predictions, 1)
        predictions_idx = tf.concat(predictions_idx, 1)
        losses = tf.concat(losses, 1)
        masks = tf.concat(masks, 1)
        
        if return_as_idx:
            return predictions_idx, total_loss, losses, masks
        else:
            return predictions, total_loss, losses, masks
   
    
    #학습 함수
    @tf.function
    def step(self, inp, targ, clipping_value = None):
        #inp.shape == (BATCH_SIZE, ENC_MAX_STEP , input_size = 2 (x,y) )
        #targ.shape == (BATCH_SIZE, DEC_MAX_STEP)
        assert (inp.shape[0] == targ.shape[0])
        loss = 0
        #local batch size
        local_bsz = inp.shape[0];
        
        with tf.GradientTape() as tape:
            predictions, loss, each_loss, mask = self.teacher_forcing_run(inp, targ)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables + [self.end_token_outputs]
        gradients = tape.gradient(loss, variables)
        if (clipping_value is not None):
            gradients, _ = tf.clip_by_global_norm(gradients, clipping_value)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss
    
    #평가하기
    def eval(self, inp, targ,
             **kwargs
            ):
        #inp.shape == (BATCH_SIZE, ENC_MAX_STEP , input_size = 2 (x,y) )
        #targ.shape == (BATCH_SIZE, DEC_MAX_STEP)
        assert (inp.shape[0] == targ.shape[0])
        loss = 0
        #local batch size
        local_bsz = inp.shape[0];
        #predictions.shape = (BATCH_SIZE, DEC_MAX_STEP, 1)
        #mask = (BATCH_SIZE, DEC_MAX_STEP)
        predictions, loss, each_loss, mask = self.naive_run(inp, targ,
                                                            return_as_idx=True)
       

        
        
        #correct_without_mask = (BATCH_SIZE, DEC_MAX_STEP)
        correct_without_mask = tf.cast(tf.equal(predictions[:, :, 0], targ), dtype=np.float32)
        step_hit = tf.reduce_sum(tf.math.multiply(correct_without_mask, mask) )
        total_step = tf.reduce_sum(mask)
        step_acc = np.array([step_hit.numpy(), total_step.numpy()], dtype = np.int32)
        
        all_step_correct = 0
        is_each_step_correct= correct_without_mask + (1.0 - mask)
        
        for i in range(local_bsz):
            if (tf.reduce_sum(is_each_step_correct[i]) >= is_each_step_correct.shape[1]):
                all_step_correct += 1;
        
        acc = all_step_correct / local_bsz
        
        ret = [predictions, acc]
        return ret
    
    
    def predict(self, inp, targ=None, **kwargs):
        # inp.shape == (BATCH_SIZE, ENC_MAX_STEP, input_size = 2 (x,y))
        if targ is not None:
            assert (inp.shape[0] == targ.shape[0])
        loss = 0
        # local batch size
        local_bsz = inp.shape[0];
        #predictions.shape = (BATCH_SIZE, DEC_MAX_STEP, 1)
        #mask = (BATCH_SIZE, DEC_MAX_STEP)
        predictions, loss, each_loss, mask = self.naive_run(inp, targ,
                                                            return_as_idx=True)
        return predictions, loss, each_loss, mask
       
        
        


