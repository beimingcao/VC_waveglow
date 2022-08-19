#!/bin/bash -e


testing_path=../current_exp

cd waveglow

for SPK_Pair in $testing_path/*/; do 
  echo $SPK_Pair
  python3 inference.py -f <(ls $SPK_Pair/RESULTS/*.pt) -w waveglow_256channels_universal_v5.pt -o ./$SPK_Pair --is_fp16 -s 0.6
done

cd ..
