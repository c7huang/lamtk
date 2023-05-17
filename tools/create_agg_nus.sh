#! /bin/bash
n_proc=3
n_seq=284
for i in $(seq $n_proc)
do
    rank=$((i-1))
    begin=$((rank*n_seq))
    end=$((begin+n_seq))
    CUDA_VISIBLE_DEVICES= nice python tools/create_complete_nus.py --begin $begin --end $end --rank $rank &
done
wait
