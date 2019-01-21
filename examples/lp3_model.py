import numpy as np
from mlstmfcn import MLSTM_FCN
from time import time
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def weasel_lp3(n_epochs = 100, batch_size = 128, 
		model_type_name = 'mlstmfcn',
		dataset_prefix = 'weasel_lp3',
		test_size = 0.2, time_stamp = None,
		verbose = False, dataset_settings = None,
		data_filename = None, Attention=True,
		Squeeze=True, return_all=False,
		use_early_stopping=False):

	time_stamp = time_stamp or int(time())

	save_filename = '{}_{}_{}_save_model_class.joblib.save'.format(
								model_type_name, dataset_prefix, time_stamp)

	data_filename = data_filename or \
						'data/WEASEL_MUSE_DATASETS/lp3/'

	dataset_settings = dataset_settings or \
						dict(n_lstm_cells = 8, 
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
							Attention = Attention, 
							Squeeze = Squeeze)

	instance = MLSTM_FCN(dataset_prefix=dataset_prefix, 
						 time_stamp=time_stamp, 
						 verbose=verbose)

	instance.load_dataset( 	data_filename = data_filename,
							normalize = False)

	instance.create_model(**dataset_settings)
	
	instance.train_model(epochs=n_epochs, 
						 batch_size=batch_size,
						 use_early_stopping=use_early_stopping)
	
	instance.save_instance(save_filename)

	instance.evaluate_model(batch_size=batch_size)
	instance.save_instance(save_filename)

	if return_all:
		return instance, features, labels, idx_train, idx_test
	else:
		return instance

instance = weasel_lp3()
