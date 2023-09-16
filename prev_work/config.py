# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications
data_folders = [
    '/home/arooba/ssd/hira/nmt dataset/nmt_scalp_eeg_dataset/normal',
    '/home/arooba/ssd/hira/nmt dataset/nmt_scalp_eeg_dataset/abnormal']
n_recordings = None  # set to an integer, if you want to restrict the set size
sensor_types = ["EEG"]
n_chans = 21
max_recording_mins = None # exclude larger recordings from training set
sec_to_cut = 60  # cut away at start of each recording
duration_recording_mins = 10  # how many minutes to use per recording
test_recording_mins = 10
max_abs_val = 8000  # for clipping#==========
sampling_freq = 200 #==========
divisor = 1  # divide signal by this#==========
test_on_eval = True  # test on evaluation set or on training set
# in case of test on eval, n_folds and i_testfold determine
# validation fold in training set for training until first stop
n_folds = 10
i_test_fold = 9
shuffle = True
model_name = 'shallow'#shallow/deep for DNN (deep terminal local 1)
n_start_chans = 25
n_chan_factor = 2  # relevant for deep model only
input_time_length = 6000
final_conv_length = 1
model_constraint = 'defaultnorm'
init_lr = 0.01#==========
batch_size = 64
max_epochs = 35 # until first stop, the continue train on train+valid
cuda = True#==========
