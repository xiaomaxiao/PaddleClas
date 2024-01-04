
python tools/infer.py -c .\ppcls\configs\PULC\car_exists\SwinTransformer_tiny_patch4_window7_224_my.yaml \
                      -o Global.device=gpu \
                      -o Global.pretrained_model=d:\Code\PaddleClas\output\SwinTransformer_tiny_patch4_window7_224\best_model \
                      -o Infer.infer_imgs=d:\Code\test\capture\chengdu-20231219-2346