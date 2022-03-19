# 19.02.22 - EffnetB0 ArcFace/ArcMargin - submit

Fix label encoder being different across CV splits.
Replace all own metric learning with Arc margin from https://www.kaggle.com/code/clemchris/pytorch-lightning-arcface-train-infer.
Add basic inference code. Needs refactoring.
Removed precision 16.
First submit.

* CV MAP: ?
* LB MAP: ?
* Commit: ?
* [https://www.kaggle.com/btseytlin/happywhale-metric-effnet-b0/edit](Kaggle notebook)
* [https://wandb.ai/btseytlin/kaggle_happywhale/runs/2m0o0xkj?workspace=user-btseytlin](Wandb run)

# 17.02.22 - EffnetB0 metric learning

Remake CV to have no test fold. Only train and val.
Add computing embedding metrics (MAP, MAPR, P@1) on val set after each validation epoch.
Add ability to train on a subset of individuals to speed up iterations.

Still can't train a single model properly, train loss does not go down, map is close to 0.

* Commit: 1e48b12d371f90093fc8610f9136f752cf3036d0
* [https://www.kaggle.com/btseytlin/happywhale-metric-effnetb-0/edit/run/90286949](Kaggle notebook)
* [https://wandb.ai/btseytlin/kaggle_happywhale/runs/1k3w86di?workspace=user-btseytlin](Wandb run)

# 14.02.22 - Baseline EffnetB0 classifier

Setup pipeline.

* Commit: b216dc56a276684344777dafb3591a19191635fd
* [https://www.kaggle.com/btseytlin/happywhale-baseline-effnetv2-s?scriptVersionId=90024302](Kaggle run)
* [https://wandb.ai/btseytlin/kaggle_happywhale/runs/3t8at480?workspace=user-btseytlin](Wandb run)
