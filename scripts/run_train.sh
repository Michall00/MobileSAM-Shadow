uv run mobile_sam/train.py  \
 --images_dir /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/images_test \
 --masks_dir /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/masks_test \
 --obj_masks_dir /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/object_masks_test \
 --epochs 1 \
 --device cpu \
 --pretrained /home/msadowski/Studia/Inzynierka/MobileSAM/weights/mobile_sam.pt \
 --wandb --wandb_mode online \
 --wandb_project mobilesam-shadow \
 --ckpt_out /home/msadowski/Studia/Inzynierka/MobileSAM/weights/mobile_sam_ckpt3.pt
