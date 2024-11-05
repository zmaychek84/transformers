#!/bin/bash


# Get folder path from command line argument, or use default if not provided
folder_path=${1:-"./results/samples_code_llama_spec_speedup_humaneval_with_iter-104000-ckpt"}
echo $folder_path

problem_file="humaneval-sub/sixty_acc_dataset.jsonl"

# Iterate through files in the folder and execute commands
for file in "$folder_path"/*.jsonl; do
    # Extract the filename
    filename=$(basename "$file")

    # Skip files with ".jsonl_results" in the filename
    if [[ $filename == *".jsonl_results"* ]]; then
        continue
    fi

    # Add prefix "Completion_" to the filename
    prefixed_filename="Completion_$filename"

    # Extract the key part needed for the command
    command_part=$(echo "$prefixed_filename" | cut -d'_' -f2-)

    # Execute the command
    # echo "evaluate_functional_correctness $folder_path/$command_part --problem_file=$problem_file"
    # If you want to actually execute the command, change the above line to:
    echo "$folder_path/$command_part"
    evaluate_functional_correctness "$folder_path/$command_part" --problem_file=$problem_file  ##> "$folder_path/pass@1_scores.txt"
done
