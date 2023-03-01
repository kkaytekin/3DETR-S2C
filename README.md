# 3DETR-S2C
<p align="center"><img src="docs/Poster.jpg"/></p>

Final report can be found [here](Report.pdf)

S2C-MMT can be found [here](https://github.com/antoniooroz/scan2cap-mmt)
### Instructions

Before running our code, the environment preperation and data installation steps provided in <a href="https://github.com/daveredrum/Scan2Cap">Scan2Cap</a> and <a href="https://github.com/facebookresearch/3detr">3DETR</a> repositories must be completed.

After the installation step, make sure to change the data-paths in `lib/config.py`. The lines needed to change are marked with `# TODO`

Important: The given model and the code base is tested in PyTorch 1.8.0. We do not guarantee compatibility with later versions.

To load the pretrained 3DETR models, download <a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep1080.pth">3DETR-m</a> and <a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep1080.pth">3DETR</a> files into `pretrained/` folder. This step can be skipped if the models trained by us are going to be loaded.

The following script can be used to reproduce our results for 3DETR-S2C:
### Confirming our Results
```
python scripts/eval.py --folder 3DETRS2C --eval_caption --min_iou [0.25 or 0.50] --eval_detection --enc_type masked --no_height --enc_dropout 0 --use_topdown --num_proposals 256 --num_locals 10 --use_relation --num_graph_steps 2
```
### Visualization
```
python scripts/visualize.py --scene_id <scene_id> --folder 3detrs2c --no_height --use_topdown --use_relation --use_orientation --num_graph_steps 2 --num_proposals 256 --num_locals 10 --enc_type masked --enc_dropout 0
```
### End-to-end Training
```
python scritps/train.py --lr 1e-3 --wd=1e-4 [--enc_type masked] --enc_dropout 0.3 --use_normal --use_topdown --num_proposals 256 --num_locals 10 --batch_size 8 --epoch 50 --use_relation --num_graph_steps 2
```
### Training with Pretrained 3DETR
```
python scripts/train_pretrained3detr.py [--unfreeze_3detr] [--enc_type masked] --lr 1e-4 --wd=1e-5 --no_height --enc_dropout 0.3 --use_topdown --num_proposals 256 --num_locals 10 --batch_size 8 --epoch 50 --use_relation --num_graph_steps 2
```
### Evaluation
```
python scripts/eval.py --folder <folder containing the model> --eval_caption --min_iou [0.25 or 0.50] [--eval_detection] [--enc_type masked] --no_height --enc_dropout 0 --use_topdown --num_proposals 256 --num_locals 10 --use_relation --num_graph_steps 2
```


# License
Scan2Cap is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

The majority of 3DETR is licensed under the Apache 2.0 license.

Please refer to [Scan2Cap](https://github.com/daveredrum/Scan2Cap) and [3DETR](https://github.com/facebookresearch/3detr) for licensing.
