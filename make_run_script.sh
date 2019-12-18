bs=1; is=128; lr=0.01; gs=8; fr=16; op=adam
datadir=orig_data
logdir=eval_log

script=run_vnet3d_with_ag.py
ct=AG_Aug1_B${bs}_I${is}_L${lr}_G${gs}_fs${fr}_${op}

cmd="python3 $script"
cmd="$cmd -ct $ct"
cmd="$cmd -I $datadir"
cmd="$cmd -bs $bs"
cmd="$cmd -is $is"
cmd="$cmd -lr $lr"
cmd="$cmd -gs $gs"
cmd="$cmd -fr $fr"
cmd="$cmd -op $op"
cmd="$cmd --n_validation 20"
cmd="$cmd --n_test 5"
#cmd="$cmd --print_summary_only"
cmd="$cmd > $logdir/${ct}.log 2>&1"

echo $cmd
eval $cmd

