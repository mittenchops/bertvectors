# conda activate bert373
export NUM_WORKER=1;
export PATH_MODEL=~/Data/cased_L-12_H-768_A-12/;
bert-serving-start -cpu -verbose -model_dir $PATH_MODEL -num_worker=$NUM_WORKER; -max_seq_len=1000
