uv run mobile_sam/test.py  \
 --images_dir /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba_test/images_test \
 --masks_dir /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba_test/masks_test \
 --obj_masks_dir /home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba_test/object_masks_test \
 --device cpu \
 --ckpt /home/msadowski/Studia/Inzynierka/MobileSAM/weights/mobile_sam_ckpt1_best.pt \
 --wandb --wandb_mode online \
 --wandb_project mobilesam-shadow-test \
