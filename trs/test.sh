

function exp(){
    ./lazyeditor.py --editing_method AlphaEdit --model_name llama3-8b --ds_range All --device 6 --label "mq_cf_sample800_2hop$1" --model_path /Data2/tangrui/EasyEdit/trs/models/LLama-3-8B-Instruct --data_json "/Data2/tangrui/EasyEdit/trs/dataset/mq_cf_sample800_2hop$1.json"
}
exp 1
exp 2
./mail.py ICT-v2 "Experiments on MQ-CF-Sample-2hop-1 and MQ-CF-Sample-2hop-2 are done."