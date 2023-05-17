#! /bin/bash
split=training
n_proc=3
n_seq=266
for i in $(seq $n_proc)
do
    rank=$((i-1))
    begin=$((rank*n_seq))
    end=$((begin+n_seq))
    echo $begin $end
    CUDA_VISIBLE_DEVICES= nice python tools/create_complete_waymo.py --split $split --begin $begin --end $end --rank $rank &
    sleep 400
done
wait

split=validation
n_proc=3
n_seq=68
for i in $(seq $n_proc)
do
    rank=$((i-1))
    begin=$((rank*n_seq))
    end=$((begin+n_seq))
    echo $begin $end
    CUDA_VISIBLE_DEVICES= nice python tools/create_complete_waymo.py --split $split --begin $begin --end $end --rank $rank &
    sleep 400
done
wait
