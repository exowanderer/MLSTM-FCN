import os
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation, Masking, Reshape
from keras.layers import multiply, concatenate
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Permute, Dropout

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

from sklearn.externals import joblib

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

class TrainValTensorboard(TensorBoard):
    # Created from
    #   https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
	def __init__(self, log_dir='./logs', **kwargs):
		
		self.log_dir = log_dir

		# Make the original `TensorBoard` log to a subdirectory 'training'
		self.training_log_dir = os.path.join(self.log_dir, 'training')
		super(TrainValTensorboard, self).__init__(self.training_log_dir, **kwargs)

		# Log the validation metrics to a separate subdirectory
		self.val_log_dir = os.path.join(log_dir, 'validation')

	def set_model(self, model):
		self.val_writer = tf.summary.FileWriter(self.val_log_dir)
		super(TrainValTensorboard, self).set_model(model)

	def on_epoch_end(self, epoch, logs=None):
		# Pop the validation logs and handle them separately with 
		#	`self.val_writer`. Also rename the keys so that they can
		#	be plotted on the same figure with the training metrics
		logs = logs or {}
		val_logs = {k.replace('val_', ''): v for k, v in logs.items() \
												if k.startswith('val_')}

		for name, value in val_logs.items():
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.val_writer.add_summary(summary, epoch)

		self.val_writer.flush()

		# Pass the remaining logs to `Tensorboard.on_epoch_end`
		logs = {k:v for k,v in logs.items() if not k.startswith('val_')}

		super(TrainValTensorboard, self).on_epoch_end(epoch, logs)

	def on_train_end(self, logs=None):
		super(TrainValTensorboard, self).on_train_end(logs)
		self.val_writer.close()

def conditional_multiple_isinstances(list_of_inputs, list_of_instances, condition='and'):
	''' Checks if a list of inputs matches a list of types
    Args:
        list_of_inputs (list, tuple): list of inputs to be checked
        list_of_instances (list, tuple): the list of instances to check if the inputs are amongst
        condition (str; optional): can be `'and'` or '`or`', 
        	depending on how the user wants to combine the booleans
	
    Returns: (bool) Condition satisfied (True) or not (False)

    Examples:
    	>>> list1 = [1,2,3,4]
    	>>> list2 = [0,2,4,8,16,32]
    	>>> tuple1 = ('thing1', 'thing2')
    	>>> array1 = np.array([list1])
    	>>> conditional_multiple_isinstances([list1, list2, list3, array1, tuple1], \
    										(list, tuple, np.array), condition='and')
    	
    	True # because all types are included in the list of instances
    	
    	>>> conditional_multiple_isinstances([list1, list2, list3, array1, tuple1], \
    										(list, tuple), condition='or')

    	True # because any of the `isinstances` are satisfied

    	>>> conditional_multiple_isinstances([list1, list2, list3, array1, tuple1], \
    										(list, tuple), condition='and')

    	False # because the array1 is not a list or tuple
    '''
	conditon_all = True

	for inputnow in list_of_inputs:
		if isinstance(inputnow, list_of_instances):
			conditon_all *= True
			if condition == 'or': return True

	return conditional_all




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

		self.trained_ = False

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
					 pre_convolve_rnn = False, 
					 pre_convolve_rnn_stride = 2, 
 					 squeeze_initializer = 'he_normal', 
					 logit_output = 'sigmoid', 
					 use_bias = False,
					 verbose = False):

		input_layer = Input(shape=self.input_shape)

		''' Create the LSTM-RNN Tower for the MLSTM-FCN Architecture '''
		if pre_convolve_rnn:
			''' sabsample timesteps to prevent "Out-of-Memory Errors" due to the Attention LSTM's size '''

			# permute to match Conv1D configuration
		    rnn_tower = Permute(permute_dims)(input_layer)
	    	rnn_tower = Conv1D(self.input_shape[0] // stride, conv1d_kernels[0], 
	    						strides=stride, padding='same', 
	    						activation=activation_func, use_bias=use_bias,
	               				kernel_initializer=local_initializer)(rnn_tower)

	        # re-permute to match LSTM configuration
	        rnn_tower = Permute(permute_dims)(rnn_tower)
			
			rnn_tower = Masking()(rnn_tower)
		else:
			# Default behaviour is to mask the input layer itself
			rnn_tower = Masking()(input_layer)

		if Attention:
			rnn_tower = AttentionLSTM(n_lstm_cells)(rnn_tower)
		else:
			rnn_tower = LSTM(n_lstm_cells)(rnn_tower)

		rnn_tower = Dropout(dropout_rate)(rnn_tower)

		''' Create the Convolution Tower for the MLSTM-FCN Architecture '''
		conv1d_tower = Permute(permute_dims)(input_layer)

		for kl, (conv1d_depth, conv1d_kernel) in enumerate(zip(conv1d_depths, conv1d_kernels)):
			# Loop over all convolution kernel sizes and depths to create the Convolution Tower
			conv1d_tower = Conv1D_Stack(conv1d_tower, 
										conv1d_depth = conv1d_depth, 
										conv1d_kernel = conv1d_kernel, 
										activation_func = activation_func, 
										local_initializer = local_initializer) 

			if Squeeze:
				conv1d_tower = squeeze_excite_block(conv1d_tower, 
											squeeze_ratio=squeeze_ratio, 
											activation_func=activation_func, 
											logit_output=logit_output, 
											kernel_initializer=squeeze_initializer,
											use_bias=use_bias)

			# Turn off Squeeze after the second to last layer
			#	to avoid Squeezing at the last layer
			if kl + 2 == len(conv1d_kernels): Squeeze = False
		
		conv1d_tower = GlobalAveragePooling1D()(conv1d_tower)

		output_layer = concatenate([rnn_tower, conv1d_tower])
		output_layer = Dense(self.num_classes, 
								activation=logit_output)(output_layer)
		
		self.model = Model(input_layer, output_layer)

		if verbose: self.model.summary()
		
		# add load model code ere to fine-tune

	def load_dataset(train_filename=None, test_filename=None,
						xtrain=None, ytrain=None, xtest=None, ytest=None,
						is_timeseries = True, normalize_timeseries=True, 
						verbose = True):

		if None in [load_train_filename, load_train_filename] \
			and None not in [xtrain, ytrain, xtest, ytest]:
			load_train_filename = load_train_filename or 'Train Data Provided Directly'
			load_test_filename = load_test_filename or 'Test Data Provided Directly'

			if conditional_multiple_isinstances((xtrain, ytrain, xtest, ytest), (list, np.array, tuple))
				self.X_train = xtrain
				self.y_train = ytrain
				self.X_test = xtest
				self.y_test = ytest
		elif conditional_multiple_isinstances((load_train_filename, load_test_filename), (str))
			if verbose: 
		    	print("Loading training data at: ", self.load_train_filename)
		    	print("Loading testing data at: ", self.load_test_filename)

		    if not os.path.exists(load_train_filename):
		        raise FileNotFoundError('File {} not found!'.format(self.load_train_filename))
			
			if not os.path.exists(load_test_filename):
		        raise FileNotFoundError('File {} not found!'.format(self.load_test_filename))

		    self.X_train, self.y_train = joblib.load(self.load_train_filename)
		    self.X_test, self.y_test = joblib.load(self.load_test_filename)
		else:
			raise ValueError("User must either provide data directly "
							 "(i.e. xtrain=ndarray, ytrain=array, ...), "
							 "\nor the file location where data is located "
							 "(i.e. train_filename = str, test_filename = str)")

		self.is_timeseries = is_timeseries
		self.normalize_timeseries = normalize_timeseries
		self.load_train_filename = load_train_filename
		self.load_test_filename = load_test_filename

	    self.num_classes = len(np.unique(self.y_train))
	    self.max_num_features = self.X_train.shape[1]
	    self.max_timesteps = self.X_train.shape[-1]
	    
		if self.normalize_timeseries: self.normalize_dataset()

	def normalize_dataset(self, x_tol=1e-8):
		self.normalize_timeseries = True # set to True because it is now

	    # scale the values
	    if self.is_timeseries:
	        X_train_mean = self.X_train.mean(axis=-1)
	        X_train_std = self.X_train.std(axis=-1)
	        self.X_train = (self.X_train - X_train_mean) / (X_train_std + x_tol)
	        self.X_test = (self.X_test - X_train_mean) / (X_train_std + x_tol)
	    else:
	    	X_train_mean = self.X_train.mean(axis=1)
	        X_train_std = self.X_train.std(axis=1)
	        self.X_train = (self.X_train - X_train_mean) / (X_train_std + x_tol)
	        self.X_test = (self.X_test - X_train_mean) / (X_train_std + x_tol)

	    if verbose: print("Finished processing train dataset..")

	    # extract labels Y and normalize to [0 - (MAX - 1)] range
	    y_train_range = (self.y_train.max() - self.y_train.min())
	    
	    self.y_train -= self.y_train.min()
	    self.y_train /= y_train_range
	    self.y_train *= self.num_classes - 1
	    
	    # extract labels Y and normalize to [0 - (MAX - 1)] range
	    # FINDME: Should we subtract the y_train.min() and divide
	    #			by the y_train_range instead of y_test?
	    y_test_range = (self.y_test.max() - self.y_test.min())
	    
	    self.y_test -= self.y_test.min()
	    self.y_test /= y_test_range
	    self.y_test *= self.num_classes - 1

        if verbose:
	        print("Finished loading test dataset..")
	        print()
	        print("Number of train samples : ", self.X_train.shape[0])
	        print("Number of test samples : ", self.X_test.shape[0])
	        print("Number of classes : ", self.num_classes)
	        print("Sequence length : ", self.X_train.shape[-1])

	def train_model(self, epochs=50, batch_size=128, val_subset=None, cutoff=None, 
						dataset_prefix='rename_me_', dataset_fold_id=None, 
						learning_rate=1e-3, monitor='loss', optimization_mode='auto', 
						compute_class_weights=True, compile_model=True,
						optimizer=None, use_model_checkpoint=True, use_lr_reduce=True,
						use_tensorboard=True, loss='categorical_crossentropy', 
						metrics=['accuracy']):

		self.learning_rate = learning_rate

	    self._classes = np.unique(self.y_train)
	    self._lbl_enc = LabelEncoder()
	    
	    y_ind = self._lbl_enc.fit_transform(self.y_train.ravel())
	    
	    if compute_class_weights:
		    recip_freq = len(self.y_train) / (len(self._lbl_enc.classes_) *
		                           np.bincount(y_ind).astype(np.float64))

		    self._class_weight = recip_freq[self._lbl_enc.transform(self.classes)]
		else:
			self._class_weight = np.ones(self.y_train.size) / self.num_classes
		
	    print("Class weights : ", self._class_weight)

	    self.y_train = to_categorical(self.y_train, len(np.unique(self.y_train)))
	    self.y_test = to_categorical(self.y_test, len(np.unique(self.y_test)))

	    if self.is_timeseries:
	        factor = 1. / np.cbrt(2)
	    else:
	        factor = 1. / np.sqrt(2)

	    if dataset_fold_id is None:
	        self.weight_fn = "./weights/{}_weights.h5".format(dataset_prefix)
	    else:
	        self.weight_fn = "./weights/{}_fold_{}_weights.h5".format(dataset_prefix, dataset_fold_id)

	    callback_list = []

	    if use_model_checkpoint:
		    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
		                                       monitor=monitor, save_best_only=True, save_weights_only=True)
		    
		    callback_list.append(model_checkpoint)

		if use_lr_reduce:
		    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
		                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)

		    callback_list.append(reduce_lr)
	    
	    if use_tensorboard:
		    tensorboard = TrainValTensorboard(  log_dir='./logs/log-{}'.format(int(time())),
		    									write_graph=False)

		   	callback_list.append(tensorboard)

		if use_early_stopping:
			early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

			callback_list.append(early_stopping)

	    if optimizer is None: optimizer = Adam(lr=self.learning_rate)

	    if compile_model:
	        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	    if val_subset is not None:
	    	# This removes 20% of the data to be ignored until after training
	    	#	should be done before this step; but it's here for completeness
	    	idx_test, idx_val = train_test_split(np.arange(y_test), test_size=0.2)
	        X_test = X_test[idx_test]
	        y_test = y_test[idx_test]

	    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
	              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))

	   	self.trained_ = True

	def evaluate_model(self, optimizer=None, test_data_subset=None, cutoff=None, dataset_fold_id=None):
		
		assert(self.trained), "Cann evaluate a model that has not been trained"

	    self.y_test = to_categorical(self.y_test, len(np.unique(self.y_test)))

	    optimizer = optimizer or Adam(lr=self.learning_rate)
	    model.compile(	optimizer=optimizer, 
	    				loss='categorical_crossentropy', metrics=['accuracy'])

	    if dataset_fold_id is None:
	        self.weight_fn = "./weights/{}_weights.h5".format(dataset_prefix)
	    else:
	        self.weight_fn = "./weights/{}_fold_{}_weights.h5".format(dataset_prefix, dataset_fold_id)

	    model.load_weights(weight_fn)

	    if test_data_subset is not None:
	        X_test = X_test[:test_data_subset]
	        y_test = y_test[:test_data_subset]

	    print("\nEvaluating : ")
	    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
	    print()
	    print("Final Accuracy : ", accuracy)

	def save_model(self, save_filename):
		joblib.dump(self, save_filename)

	def load_model(self, load_filename):
		self.__dict__ = joblib.load(load_filename).__dict__ 

if __name__ == "__main__":

	model_type_name = 'mlstm_fcn'
	data_set_name = 'plasticc'
	
	save_filename = '{}_{}_{}_save_model_class.joblib.save'.format(model_type_name, data_set_name, int(time()))

	dataset_settings = dict(n_lstm_cells = 8, 
							dropout_rate = 0.8, 
							permute_dims = (2,1), 
							conv1d_depths = [128, 256, 128], 
							conv1d_kernels = [8, 5, 3], 
							local_initializer = 'he_uniform', 
							activation_func = 'relu', 
							squeeze_ratio = 16, 
							logit_output = 'sigmoid', 
							squeeze_initializer = 'he_normal', 
							use_bias = False, 
							verbose = False)
			 				# Attention = False, # Default
			 				# Squeeze = True,  # Default
	
	# Model 1
	instance1 = MLSTM_FCN(DATASET_INDEX=DATASET_INDEX)

	instance1.create_model(**dataset_settings)
	instance1.load_dataset(load_train_filename, load_test_filename, normalize_timeseries=True)

	instance1.train_model(epochs=1000, batch_size=128)
	instance1.evaluate_model(batch_size=128)
	instance1.save_model(save_filename)
	# # Model 2
	# instance2 = MLSTM_FCN(DATASET_INDEX=DATASET_INDEX)
	# instance2.create_model(Attention=True, **dataset_settings)

	# # Model 3
	# instance3 = MLSTM_FCN(DATASET_INDEX=DATASET_INDEX)
	# instance3.create_model(Squeeze=False, **dataset_settings)

	# # Model 4
	# instance4 = MLSTM_FCN(DATASET_INDEX=DATASET_INDEX)
	# instance4.create_model(Attention=True, Squeeze=False, **dataset_settings)

	# train_model(instance1.model, X_train, y_train, X_test, y_test, is_timeseries, epochs=1000, batch_size=128)
	# evaluate_model(instance1.model, X_test, y_test, is_timeseries, batch_size=128)
