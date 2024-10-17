fold=$1
output_dir='../output/pascal_prompt_self'



python evaluate_segmentation_pascal.py \
        --output_dir ${output_dir}/${fold} \
        --fold ${fold} \
        --search \
        --ensemble
