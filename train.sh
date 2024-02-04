
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name="ABSA_BERT" \
    --fix_tfm="partial" \
    --absa_type="gru" \
    --num_epochs=14