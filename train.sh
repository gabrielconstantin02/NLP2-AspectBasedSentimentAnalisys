
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name="ABSA_BERT" \
    --transformer_name='bert-base-uncased' \
    --fix_tfm="partial" \
    --absa_type="gru" \
    --data_path='data/rest' \
    --num_epochs=9 \
    --batch_size=8