python mobile_sam/prepare_coco.py \
  --lvis-ann /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/COCO/annotations/lvis_v1_val.json \
  --img-root /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/COCO/val2017/ \
  --bg-root /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/ISTD_Dataset/train/train_C/ \
  --out-root ./synthetic_no_shadow \
  --n 10 \
  --allowed-category-ids 1 \
  --min-mask-area 5000 \