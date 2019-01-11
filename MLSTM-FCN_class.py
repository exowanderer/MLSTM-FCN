from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation, Masking, Reshape
from keras.layers import multiply, concatenate
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Permute, Dropout

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

def Conv1D_Stack(input_stack, conv1d_depth, conv1d_kernel, 
				activation_func, local_initializer):
	
	output_stack = Conv1D(conv1d_depth, conv1d_kernel,
						kernel_initializer=local_initializer)(input_stack)

	output_stack = BatchNormalization()(output_stack)
	
	output_stack = Activation(activation_func)(output_stack)

	return output_stack

def squeeze_excite_block(tower, 
						 squeeze_ratio = 16, 
						 activation_func = 'relu',
						 logit_output = 'sigmoid', 
						 kernel_initializer = 'he_normal', 
						 use_bias = False):
    ''' Create a squeeze-excite block
    Args:
        tower: tower tensor
        filters: number of output filters
        k: width factor
	
    Returns: a keras tensor
    '''
    filters = tower._keras_shape[-1] # channel_axis = -1 for TF

    squeeze_excite = GlobalAveragePooling1D()(tower)
    squeeze_excite = Reshape((1, filters))(squeeze_excite)

    squeeze_excite = Dense(filters // squeeze_ratio, 
    					activation=activation_func,
    					kernel_initializer=kernel_initializer, 
    					use_bias=use_bias)(squeeze_excite)
    squeeze_excite = Dense(filters, activation=logit_output, 
    			kernel_initializer=kernel_initializer, 
    			use_bias=use_bias)(squeeze_excite)
    
    squeeze_excite = multiply([tower, squeeze_excite])
    
    return squeeze_excite

class MLSTM_FCN(object):
	def __init__(self, DATASET_INDEX=None, TRAINABLE=True):

		num_max_times = MAX_TIMESTEPS_LIST[DATASET_INDEX]
		num_max_var = MAX_NB_VARIABLES[DATASET_INDEX]
		
		self.num_classes = NB_CLASSES_LIST[DATASET_INDEX]

		self.input_shape = (num_max_var, num_max_times)

	def create_model(self, 
					 n_lstm_cells = 8, 
					 dropout_rate = 0.8, 
					 permute_dims = (2,1), 
					 conv1d_depths = [128, 256, 128], 
					 conv1d_kernels = [8, 5, 3], 
					 local_initializer = 'he_uniform', 
					 activation_func = 'relu', 
					 Attention = False, 
 					 Squeeze = True, 
					 squeeze_ratio = 16, 
 					 squeeze_initializer = 'he_normal', 
					 logit_output = 'sigmoid', 
					 use_bias = False):

		input_layer = Input(shape=self.input_shape)

		rnn_tower = Masking()(input_layer)

		if Attention:
			rnn_tower = AttentionLSTM(n_lstm_cells)(rnn_tower)
		else:
			rnn_tower = LSTM(n_lstm_cells)(rnn_tower)

		rnn_tower = Dropout(dropout_rate)(rnn_tower)

		conv1d_tower = Permute(permute_dims)(input_layer)

		# conv1d_tower = Conv1D(conv1d_depths[0], conv1d_kernels[0],
		# 					kernel_initializer=local_initializer)(conv1d_tower)
		# conv1d_tower = BatchNormalization()(conv1d_tower)
		# conv1d_tower = Activation(activation_func)(conv1d_tower)
		conv1d_tower = Conv1D_Stack(conv1d_tower, 
									conv1d_depth = conv1d_depths[0], 
									conv1d_kernel = conv1d_kernels[0], 
									activation_func = activation_func, 
									local_initializer = local_initializer) 

		if Squeeze:
			conv1d_tower = squeeze_excite_block(conv1d_tower, 
										squeeze_ratio=squeeze_ratio, 
										activation_func=activation_func, 
										logit_output=logit_output, 
										kernel_initializer=squeeze_initializer,
										use_bias=use_bias)
		
		# conv1d_tower = Conv1D(conv1d_depths[1], conv1d_kernels[1],
		# 					kernel_initializer=local_initializer)(conv1d_tower)
		# conv1d_tower = BatchNormalization()(conv1d_tower)
		# conv1d_tower = Activation(activation_func)(conv1d_tower)
		conv1d_tower = Conv1D_Stack(conv1d_tower, 
									conv1d_depth = conv1d_depths[1], 
									conv1d_kernel = conv1d_kernels[1], 
									activation_func = activation_func, 
									local_initializer = local_initializer) 
		
		if Squeeze:
			conv1d_tower = squeeze_excite_block(conv1d_tower, 
										squeeze_ratio=squeeze_ratio, 
										activation_func=activation_func, 
										logit_output=logit_output, 
										kernel_initializer=squeeze_initializer,
										use_bias=use_bias)
		
		# conv1d_tower = Conv1D(conv1d_depths[2], conv1d_kernels[2],
		# 					kernel_initializer=local_initializer)(conv1d_tower)
		# conv1d_tower = BatchNormalization()(conv1d_tower)
		# conv1d_tower = Activation(activation_func)(conv1d_tower)
		conv1d_tower = Conv1D_Stack(conv1d_tower, 
									conv1d_depth = conv1d_depths[2], 
									conv1d_kernel = conv1d_kernels[2], 
									activation_func = activation_func, 
									local_initializer = local_initializer) 

		conv1d_tower = GlobalAveragePooling1D()(conv1d_tower)

		output_layer = concatenate([rnn_tower, conv1d_tower])
		output_layer = Dense(self.num_classes, 
								activation=logit_output)(output_layer)
		
		self.model = Model(input_layer, output_layer)

		self.model.summary()
		
		# add load model code ere to fine-tune


if __name__ == "__main__":

	arabic_voice_DATASET_INDEX = 27
	arabic_voice_settings = dict(n_lstm_cells = 8, 
								 dropout_rate = 0.8, 
								 permute_dims = (2,1), 
								 conv1d_depths = [128, 256, 128], 
								 conv1d_kernels = [8, 5, 3], 
								 local_initializer = 'he_uniform', 
								 activation_func = 'relu', 
								 squeeze_ratio = 16, 
								 logit_output = 'sigmoid', 
								 squeeze_initializer = 'he_normal', 
								 use_bias = False)
			 					 # Attention = False, 
			 					 # Squeeze = True, 
    # Model 1
    instance1 = MLSTM_FCN(DATASET_INDEX=arabic_voice_DATASET_INDEX)
    instance1.create_model(**arabic_voice_settings)

    # Model 2
    instance2 = MLSTM_FCN(DATASET_INDEX=arabic_voice_DATASET_INDEX)
    instance2.create_model(Attention=True, **arabic_voice_settings)
    
    # Model 3
    instance3 = MLSTM_FCN(DATASET_INDEX=arabic_voice_DATASET_INDEX)
    instance3.create_model(Squeeze=False, **arabic_voice_settings)
    
    # Model 4
    instance4 = MLSTM_FCN(DATASET_INDEX=arabic_voice_DATASET_INDEX)
    instance4.create_model(Attention=True, Squeeze=False, **arabic_voice_settings)
    
    train_model(instance1.model, DATASET_INDEX, dataset_prefix='arabic_voice_', epochs=1000, batch_size=128)
    evaluate_model(instance1.model, DATASET_INDEX, dataset_prefix='arabic_voice_', batch_size=128)
