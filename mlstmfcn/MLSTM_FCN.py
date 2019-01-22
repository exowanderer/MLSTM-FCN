import os
import tensorflow as tf
import numpy as np

from .utils.layer_utils import AttentionLSTM

from keras.layers import Input, Dense, LSTM, Activation, Masking, Reshape
from keras.layers import multiply, concatenate
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Permute, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard, EarlyStopping

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

from time import time

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

class TrainValTensorboard(TensorBoard):
	# Created from
	#   https://stackoverflow.com/questions/47877475/
	#	keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
	def __init__(self, log_dir='./logs', **kwargs):
		
		self.log_dir = log_dir

		# Make the original `TensorBoard` log to a subdirectory 'training'
		self.training_log_dir = os.path.join(self.log_dir, 'training')
		super(TrainValTensorboard, self).__init__(self.training_log_dir, 
														**kwargs)

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

def isinstances(list_of_inputs, list_of_instances, condition='and'):
	''' Checks if a list of inputs matches a list of types
	Args:
		list_of_inputs (list, tuple): list of inputs to be checked
		list_of_instances (list, tuple): the list of instances 
			to check if the inputs are amongst
		condition (str; optional): can be `'and'` or '`or`', 
			depending on how the user wants to combine the booleans
	
	Returns: (bool) Condition satisfied (True) or not (False)

	Examples:
		>>> list1 = [1,2,3,4]
		>>> list2 = [0,2,4,8,16,32]
		>>> tuple1 = ('thing1', 'thing2')
		>>> array1 = np.array([list1])
		>>> isinstances([list1, list2, list3, array1, tuple1], \
									(list, tuple, np.array), condition='and')
		
		True # because all types are included in the list of instances
		
		>>> isinstances([list1, list2, list3, array1, tuple1], \
											(list, tuple), condition='or')

		True # because any of the `isinstances` are satisfied

		>>> isinstances([list1, list2, list3, array1, tuple1], \
											(list, tuple), condition='and')

		False # because the array1 is not a list or tuple
	'''
	conditon_all = True
	
	for inputnow in list_of_inputs:
		if isinstance(inputnow, list_of_instances):
			conditon_all *= True
			if condition == 'or': return True

	return conditon_all

def load_data_switch(data_filename):
	if data_filename[-1] == '/': 
		X_train = np.load(data_filename+'X_train.npy')
		y_train = np.load(data_filename+'y_train.npy')

		X_test = np.load(data_filename+'X_test.npy')
		y_test = np.load(data_filename+'y_test.npy')
	elif '.joblib.save' in data_filename:
		features, labels = joblib.load(data_filename)
		idx_train, idx_test = train_test_split(np.arange(len(labels)), 
												test_size = test_size)
		X_train = features[idx_train]
		y_train = labels[idx_train]

		X_test = features[idx_test]
		y_train = labels[idx_test]
	elif '.npy' in data_filename:
		'''
			Example: if data_filane = 'X_train.npy'
			then data_filename.replace('y','X').replace('test','train') 
			is irrelevant; but data_filename.replace('y','X').replace('train','test')
			is critical. There is always one out of four that will be irrlevant

		'''
		X_train = np.load(data_filename.replace('y','X').replace('test','train'))
		y_train = np.load(data_filename.replace('X','y').replace('test','train'))

		X_test = np.load(data_filename.replace('y','X').replace('train','test'))
		y_test = np.load(data_filename.replace('X','y').replace('train','test'))
	else:
		raise ValueError("`data_filename` must end in either a directory `'/'`"
							" or `.npy` or `.joblib.save`"
							"\nIf ending in directory, then files X_train.npy,"
							" y_train.npy, X_test.npy, y_train.npy must exist"
							" in that directory."
							"\nIf ending in `.joblib.save`, then data file "
							" will be loaded as features, labels joblib.load()"
							"\nIf ending in `.npy`, then files X_train.npy,"
							" y_train.npy, X_test.npy, y_train.npy are all"
							" assumed to be in the same directory")
	
	return X_train, y_train, X_test, y_test

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
	def __init__(self, dataset_prefix='rename_me_', time_stamp=int(time()), 
						verbose=False):

		self.verbose = verbose
		self.time_stamp = time_stamp
		self.dataset_prefix = dataset_prefix 
		self.dataset_fold_id = None
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
			''' sabsample timesteps to prevent "Out-of-Memory Errors" 
					due to the Attention LSTM's size '''

			# permute to match Conv1D configuration
			rnn_tower = Permute(permute_dims)(input_layer)
			rnn_tower = Conv1D(self.input_shape[0]//stride, conv1d_kernels[0], 
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

		zipper = zip(conv1d_depths, conv1d_kernels)
		for kl, (conv1d_depth, conv1d_kernel) in enumerate(zipper):
			# Loop over all convolution kernel sizes and depths 
			#	to create the Convolution Tower
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

		if self.verbose or verbose: self.model.summary()
		
		# add load model code ere to fine-tune

	def load_dataset(self, data_filename=None, 
						xtrain=None, ytrain=None, xtest=None, ytest=None,
						is_timeseries = True, normalize=True, 
						weights_dir = './weights/', compute_class_weights=True,
						verbose = False):

		# Check if data is provided directly: i.e. no `None`s
		data_chk = np.all([t is not None for t in [xtrain,ytrain,xtest,ytest]])

		if data_filename is None and data_chk:
			data_filename = data_filename or 'Train Data Provided Directly'

			if isinstances((xtrain,ytrain,xtest,ytest), \
								(list,np.ndarray,tuple)):
				self.X_train = xtrain
				self.y_train = ytrain
				self.X_test = xtest
				self.y_test = ytest
		elif isinstance(data_filename, (str)):
			if self.verbose or verbose:
				print("Loading data from: ", self.data_filename)

			if not os.path.exists(data_filename):
				raise FileNotFoundError('File {} not found!'.format(\
											data_filename))
			
			self.X_train, self.y_train, self.X_test, self.y_test = \
												load_data_switch(data_filename)
		else:
			raise ValueError("User must either provide data directly "
							"(i.e. xtrain=ndarray, ytrain=array, ...), "
							"\nor the file location where data is located "
							"(i.e. data_filename = str)")

		self._LabelEncoder = LabelEncoder()
		self.y_train = self._LabelEncoder.fit_transform(self.y_train)
		self.y_test = self._LabelEncoder.transform(self.y_test)

		self.is_timeseries = is_timeseries
		self.normalize = normalize
		self.data_filename = data_filename

		self.classes = np.unique(self.y_train)
		self.num_classes = len(self.classes)
		self.num_samples = len(self.y_train)
		self.num_features = self.X_train.shape[1]
		self.num_timesteps = self.X_train.shape[-1]
		
		self.input_shape = (self.num_features, self.num_timesteps)

		if self.normalize: self.normalize_dataset()

		self.y_train = to_categorical(self.y_train, self.num_classes)
		self.y_test = to_categorical(self.y_test, self.num_classes)

		if compute_class_weights:
			# len_lbl_enc = len(self._LabelEncoder.classes_)
			sum_y_train = self.y_train.sum(axis=0)
			
			recip_freq = self.num_samples / (self.num_classes * sum_y_train)
			self.class_weights_ = recip_freq
			#[self._LabelEncoder.transform(self.classes)]
		else:
			self.class_weights_ = np.ones(self.y_train.size) / self.num_classes

		if verbose or self.verbose: 
			print("Class weights : ", self.class_weights_)

		if self.dataset_fold_id is None:
			self.weight_fn = weights_dir + "{}_{}_weights.h5".format(
									self.dataset_prefix, self.time_stamp)
		else:
			self.weight_fn = weights_dir + \
								"{}_{}_fold_{}_weights.h5".format(
								self.dataset_prefix, self.time_stamp, 
								self.dataset_fold_id)

		while os.path.exists(self.weight_fn):
			if self.dataset_fold_id is None: self.dataset_fold_id = 1

			if 'fold' in self.weight_fn:
				old_fold = '_fold_{}_weights.h5'.format(self.dataset_fold_id-1)
			else:
				old_fold = '_weights.h5'

			new_fold = '_fold_{}_weights.h5'.format(self.dataset_fold_id)

			self.weight_fn = self.weight_fn.replace(old_fold, new_fold)

			self.dataset_fold_id += 1

	def normalize_dataset(self, x_tol=1e-8, normalize_labels=False, 
							verbose=False):
		
		self.normalize = True # set to True because it is now

		# scale the values
		if self.is_timeseries:
			X_train_mean = self.X_train.mean(axis=0)
			X_train_std = self.X_train.std(axis=0)
			self.X_train = (self.X_train - X_train_mean) / (X_train_std+x_tol)
			self.X_test = (self.X_test - X_train_mean) / (X_train_std+x_tol)
		else:
			X_train_mean = self.X_train.mean(axis=0)
			X_train_std = self.X_train.std(axis=0)
			self.X_train = (self.X_train - X_train_mean) / (X_train_std+x_tol)
			self.X_test = (self.X_test - X_train_mean) / (X_train_std+x_tol)

		if self.verbose or verbose: 
			print("Finished processing train dataset..")

		if normalize_labels:
			# extract labels Y and normalize to [0 - (MAX - 1)] range
			self._y_train_min = self.y_train.min()
			self._y_train_max = self.y_train.max()
			self._y_train_range = (self._y_train_max - self._y_train_min)
			
			self.y_train = self.y_train - self._y_train_min
			self.y_train = self.y_train / self._y_train_range
			self.y_train = self.y_train * (self.num_classes - 1)
			
			# extract labels Y and normalize to [0 - (MAX - 1)] range
			# FINDME: Should we subtract the y_train.min() and divide
			#			by the y_train_range instead of y_test?
			self._y_test_min = self.y_test.min()
			self._y_test_max = self.y_test.max()
			self._y_test_range = (self._y_test_max - self._y_test_min)
			
			self.y_test = self.y_test - self._y_test_min
			self.y_test = self.y_test / self._y_test_range
			self.y_test = self.y_test * (self.num_classes - 1)

		if self.verbose or verbose:
			print("Finished loading test dataset..")
			print()
			print("Number of train samples : ", self.X_train.shape[0])
			print("Number of test samples : ", self.X_test.shape[0])
			print("Number of classes : ", self.num_classes)
			print("Number of features : ", self.num_features)
			print("Sequence length : ", self.num_timesteps)

	def train_model(self, epochs=50, batch_size=128, val_subset_size=None, 
						cutoff=None, dataset_fold_id=None, learning_rate=1e-3, 
						monitor='loss', optimization_mode='auto', 
						compile_model=True, optimizer=None, 
						use_model_checkpoint=True, use_lr_reduce=True, 
						use_tensorboard=True, use_early_stopping=True, 
						metrics=['accuracy'], loss='categorical_crossentropy',
						logdir = './logs/log-{}', callback_list=None):

		self.dataset_fold_id = dataset_fold_id

		if '{}' in logdir: 
			logdir = logdir.format(self.time_stamp)

		self.learning_rate = learning_rate

		# y_ind = self._LabelEncoder.transform(self.y_train.ravel())
		
		if self.is_timeseries:
			factor = 1. / np.cbrt(2)
		else:
			factor = 1. / np.sqrt(2)

		callback_list = callback_list or []

		if use_model_checkpoint:
			abs_weight_path = os.path.dirname(os.path.abspath(self.weight_fn))
			if not os.path.exists(abs_weight_path):
				print("Creating {}".format(abs_weight_path))
				os.mkdir(abs_weight_path)

			model_checkpoint = ModelCheckpoint(self.weight_fn, 
												verbose=1, 
												mode=optimization_mode, 
												monitor=monitor, 
												save_best_only=True, 
												save_weights_only=True)
			
			callback_list.append(model_checkpoint)

		if use_lr_reduce:
			reduce_lr = ReduceLROnPlateau(	monitor=monitor, 
											patience=100, 
											mode=optimization_mode,
											factor=factor, 
											cooldown=0, 
											min_lr=1e-4, 
											verbose=2)

			callback_list.append(reduce_lr)
		
		if use_tensorboard:
			tensorboard = TrainValTensorboard(log_dir=logdir,write_graph=False)

			callback_list.append(tensorboard)

		if use_early_stopping:
			early_stopping = EarlyStopping(monitor='val_acc', min_delta=0,
											patience=10, verbose=1, 
											mode='auto')

			callback_list.append(early_stopping)

		if optimizer is None: optimizer = Adam(lr=self.learning_rate)

		if compile_model:
			self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

		if val_subset_size is not None:
			# This removes 20% of the data to be ignored until after training
			#	should be done before this step; but it's here for completeness
			y_test_idx = np.arange(self.y_test.size)
			idx_test, idx_val = train_test_split(y_test_idx, 
												test_size=val_subset_size)

			X_test_local = self.X_test[idx_test]
			y_test_local = self.y_test[idx_test]
		else:
			X_test_local = self.X_test
			y_test_local = self.y_test

		self.results_ = self.model.fit(self.X_train, self.y_train, 
										batch_size = batch_size, 
										epochs = epochs, 
										callbacks = callback_list, 
										verbose = 2, 
										class_weight = self.class_weights_, 
										validation_data = (X_test_local, 
															y_test_local))

		self.trained_ = True

	def evaluate_model(self, ytest=None, batch_size=128, optimizer=None, 
						test_subset_size=None, cutoff=None, 
						dataset_fold_id=None, return_all=False):
		
		assert(self.trained_), "Cannot evaluate a model "\
							   "that has not been trained"

		if ytest is not None:
			self.y_test = ytest

			if self.normalize_dataset:
				self.y_test = self.y_test - self._y_test_min
				self.y_test = self.y_test / self._y_test_range
				self.y_test = self.y_test * (self.num_classes - 1)

			self.y_test = to_categorical(self.y_test, self.num_classes)

		optimizer = optimizer or Adam(lr=self.learning_rate)
		self.model.compile(	optimizer=optimizer, 
						loss='categorical_crossentropy', 
						metrics=['accuracy'])

		self.model.load_weights(self.weight_fn)

		if test_subset_size is not None:
			# This removes 20% of the data to be ignored until after training
			#	should be done before this step; but it's here for completeness
			y_test_idx = np.arange(self.y_test.size)
			idx_test, idx_val = train_test_split(y_test_idx, 
												test_size=test_subset_size)

			X_test_local = self.X_test[idx_test]
			y_test_local = self.y_test[idx_test]
		else:
			X_test_local = self.X_test
			y_test_local = self.y_test

		print("\nEvaluating : ")
		loss, accuracy = self.model.evaluate(self.X_test, self.y_test, 
											batch_size=batch_size)
		print("\nFinal Loss : {}\nFinal Accuracy : {}".format(loss, accuracy))

		if return_all: return loss, accuracy

	def predict(self):
		assert(self.trained_), "Cannot evaluate a model "\
							   "that has not been trained"

		if ytest is not None:
			self.y_test = ytest

			if self.normalize_dataset:
				self.y_test = self.y_test - self._y_test_min
				self.y_test = self.y_test / self._y_test_range
				self.y_test = self.y_test * (self.num_classes - 1)

			self.y_test = to_categorical(self.y_test, self.num_classes)

		optimizer = optimizer or Adam(lr=self.learning_rate)
		self.model.compile(	optimizer=optimizer, 
						loss='categorical_crossentropy', 
						metrics=['accuracy'])

		self.model.load_weights(self.weight_fn)

		if test_subset_size is not None:
			# This removes 20% of the data to be ignored until after training
			#	should be done before this step; but it's here for completeness
			y_test_idx = np.arange(self.y_test.size)
			idx_test, idx_val = train_test_split(y_test_idx, 
												test_size=test_subset_size)

			X_test_local = self.X_test[idx_test]
			y_test_local = self.y_test[idx_test]
		else:
			X_test_local = self.X_test
			y_test_local = self.y_test

		if self.verbose or verbose: print("\n[INFO] Predicting: ")
		
		prediction = self.model.predict(self.X_test)

		prediction = prediction.argmax(axis=1)
		prediction = self._LabelEncoder.inverse_transform(prediction)

		return prediction

	def save_instance(self, save_filename, verbose=False):
		if self.verbose or verbose:
			print('[INFO]: Saving Results to {}'.format(save_filename))

		self.save_filename = save_filename

		joblib.dump(self.__dict__ , self.save_filename)

	def load_instance(self, load_filename):
		self.__dict__ = joblib.load(load_filename)

def main(model_type_name='', dataset_prefix='', n_samples=100, 
			n_features=20, n_timesteps=25, verbose=False,
			n_classes=20, n_possible_classes=100, test_size=0.2,
			x_mean=10, x_std=3, classes=[0,1,2,3], n_epochs=2, batch_size=128,
			save_dir='./', time_stamp=None, seed=42, return_all=True):

	import numpy as np

	from mlstmfcn import MLSTM_FCN
	from time import time
	from sklearn.model_selection import train_test_split

	np.random.seed(seed)
	classes = np.random.choice(np.arange(n_possible_classes), size=n_classes)
	classes = np.array(['class{}'.format(c) for c in classes.astype(str)])

	features = np.random.normal(x_mean, x_std, 
				(n_samples, n_features, n_timesteps))
	
	labels = np.random.choice(classes, features.shape[0])
	
	idx_train, idx_test = train_test_split(np.arange(len(labels)), 
											test_size = test_size)
		
	time_stamp = time_stamp or int(time())
	
	save_filename = save_dir + '{}_{}_{}_save_model_class.joblib.save'.format(
								model_type_name, dataset_prefix, time_stamp)

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
							verbose = verbose,
							Attention = False, 
							Squeeze = True)

	# Model 1
	instance = MLSTM_FCN(dataset_prefix=dataset_prefix, 
						 time_stamp = time_stamp,
						 verbose = verbose)

	instance.load_dataset( 	xtrain = features[idx_train], 
							xtest = features[idx_test], 
							ytrain = labels[idx_train], 
							ytest = labels[idx_test], 
							normalize = True,
							compute_class_weights=True)
	return instance, features, labels, idx_train, idx_test
	instance.create_model(**dataset_settings)
	
	instance.train_model(epochs=n_epochs, batch_size=batch_size)
	instance.save_instance(save_filename=save_filename)	

	instance.evaluate_model(batch_size=batch_size)

	if return_all:
		return instance, features, labels, idx_train, idx_test
	else:
		return instance

if __name__ == '__main__':
	from mlstmfcn import MLSTM_FCN, main
	from time import time

	model_type_name = 'mlstm_fcn'
	dataset_prefix = 'plasticc'
	batch_size = 128

	instance1 = main(model_type_name, 
					 dataset_prefix=dataset_prefix, 
					 verbose=True)

	instance2 = MLSTM_FCN(verbose=True)
	instance2.load_instance(instance1.save_filename)

	instance2.evaluate_model(batch_size=batch_size)