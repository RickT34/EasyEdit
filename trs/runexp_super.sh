#!/bin/bash
function run(){
./runexp.sh $1 /Data2/tangrui/EasyEdit/trs/dataset/mq_cf_sample800_2hop1.json mq_cf_sample800_2hop1
./runexp.sh $1 /Data2/tangrui/EasyEdit/trs/dataset/mq_cf_sample800_2hop2.json mq_cf_sample800_2hop2
}
run AlphaEdit
run FT-L
run FT-M
run DPO
run LoRA
run QLoRA
run ROME
run UltraEdit
run UnKE
# run MEND
# run MEMIT
# run LoRA
# run SERAC
# run Grace
./mail.py "ICT-v2" "All Done"