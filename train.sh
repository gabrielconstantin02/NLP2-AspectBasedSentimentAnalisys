
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name="ABSA_BERT" \
    --transformer_name='roberta-base' \
    --fix_tfm="partial" \
    --absa_type="gru" \
    --data_path='data/laptop14' \
    --num_epochs=14 \
    --batch_size=25
    # --transformer_name='bert-base-uncased' \
    # --data_path='data/rest' \
    # --num_epochs=9 \
    # --batch_size=16