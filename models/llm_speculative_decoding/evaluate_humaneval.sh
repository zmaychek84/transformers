export CUDA_VISIBLE_DEVICES="2,3"
TARGET_MODEL_PATH="/group/modelzoo/sequence_learning/weights/nlp-pretrained-model/llama_split/code-llama-2-7b"
DRAFT_MODEL_PATH="./models/iter-104000-ckpt"
TEMPERATURE=0.1
MAX_STEP_DRAFT=10
# HUMANEVAL_PATH="./openai_humaneval_dataset"   ##"./humaneval-sub/sixty_acc_dataset.json"
HUMANEVAL_PATH="./humaneval-sub/sixty_acc_dataset.json"

##tmp: run pipeline
python evaluate_humaneval.py --target_model=$TARGET_MODEL_PATH \
                             --draft_model=$DRAFT_MODEL_PATH \
                             --temperature=$TEMPERATURE \
                             --max_step_draft=$MAX_STEP_DRAFT \
                             --humanevalpath=$HUMANEVAL_PATH
