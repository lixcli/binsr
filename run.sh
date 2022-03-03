DATA_DIR=/media/DATA/SR

edsr_baseline_x() {
python main_ori.py --model edsr --scale $1 \
--save edsr_baseline_x$1 --reset \
--patch_size $2 \
--epochs 300 \
--decay 200 \
--gclip 0 \
--dir_data $DATA_DIR
}
# edsrbasline
# CUDA_VISIBLE_DEVICES=1 edsr_baseline_x 4 192
# CUDA_VISIBLE_DEVICES=3 edsr_baseline_x 2 96


########### eval #################
edsrx2_noskt_eval() {
python3 main_noskt.py --scale 2 --model edsr \
--k_bits 1 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--model_name bin \
--save "../binsr_experiment/edsr/bin_noskt_x2/1bit" --dir_data $DATA_DIR 
}

edsrx2_baseline_noskt_eval() {
python3 main_ori.py --scale 2 --model edsr \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine "pretrained/edsr_baseline_x2.pt"
--dir_data $DATA_DIR 
}

edsrx4_baseline_noskt_eval() {
python3 main_ori.py --scale 4 --model edsr \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine "pretrained/edsr_baseline_x4.pt"
--dir_data $DATA_DIR 
}

# edsrx2_noskt_eval

edsrx4_noskt_eval() {
python3 main_noskt.py --scale 4 --model edsr \
--k_bits 1 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--model_name bin \
--save "../binsr_experiment/edsr/bin_noskt_x4/1bit" --dir_data $DATA_DIR 
}
# edsrx4_noskt_eval
srrx2_noskt_eval() {
python3 main_noskt.py --scale 2 --model srresnet \
--k_bits 1 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--model_name bin \
--save "../binsr_experiment/srresnet/bin_noskt_x2/1bit" --dir_data $DATA_DIR 
}

# srrx2_noskt_eval

srrx4_noskt_eval() {
python3 main_noskt.py --scale 4 --model srresnet \
--k_bits 1 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--model_name bin \
--save "../binsr_experiment/srresnet/bin_noskt_x4/1bit" --dir_data $DATA_DIR 
}

srrx4_noskt_eval

# edsrx4_noskt_eval
########### noskt ################

bin_edsr_noskt_x4() {
python main_noskt.py --scale 4 \
--k_bits 1 --model EDSR \
--patch_size 192 \
--data_test Set14 \
--model_name $1 \
--reset \
--epochs 300 \
--decay 60 \
--save "edsr/$1_noskt_x4/1bit" --dir_data $DATA_DIR --print_every 10
}

bin_edsr_noskt_x2() {
python main_noskt.py --scale 2 \
--k_bits 1 --model EDSR \
--patch_size 96 \
--data_test Set14 \
--model_name $1 \
--epochs 300 \
--decay 60 \
--reset \
--save "edsr/$1_noskt_x2/1bit" --dir_data $DATA_DIR --print_every 10
}

bin_srresnet_noskt_x2() {
python main_noskt.py --scale 2 \
--k_bits 1 --model srresnet \
--patch_size 96 \
--data_test Set14 \
--model_name $1 \
--epochs 300 \
--decay 60 \
--reset \
--save "srresnet/$1_noskt_x2/1bit" --dir_data $DATA_DIR --print_every 10
}

bin_srresnet_noskt_x4() {
python main_noskt.py --scale 4 \
--k_bits 1 --model srresnet \
--patch_size 192 \
--data_test Set14 \
--model_name $1 \
--epochs 300 \
--decay 60 \
--reset \
--save "srresnet/$1_noskt_x4/1bit" --dir_data $DATA_DIR --print_every 10
}
# CUDA_VISIBLE_DEVICES=2 bin_edsr_noskt_x4 bin
# CUDA_VISIBLE_DEVICES=2 bin_edsr_noskt_x2 bin

########### skt ##################

binedsr_x4() {
python main.py --scale 4 \
--k_bits 1 --model EDSR \
--pre_train ./pretrained/edsr_baseline_x4.pt --patch_size 192 \
--data_test Set14 \
--model_name $1 \
--save "output/$1_edsrx4/1bit" --dir_data $DATA_DIR --print_every 10
}

# CUDA_VISIBLE_DEVICES=1 binedsr_x4 bin

########### srresnet ##############

srresnet_baseline_x4() {
python main_ori.py --model SRResnet --scale 4 \
--save srresnet_baseline_x4 --reset \
--patch_size 192 \
--epochs 500 \
--decay 200 \
--gclip 0 \
--dir_data $DATA_DIR
}

# CUDA_VISIBLE_DEVICES=2 srresnet_baseline_x4

# CUDA_VISIBLE_DEVICES=1 bin_srresnet_noskt_x2 bin
# CUDA_VISIBLE_DEVICES=1 bin_srresnet_noskt_x4 bin

# CUDA_VISIBLE_DEVICES=0 bin_edsr_noskt_x2 bin
# CUDA_VISIBLE_DEVICES=0 bin_edsr_noskt_x4 bin
