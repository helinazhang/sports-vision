#!/usr/bin/env python3
# -- coding:utf-8 --
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import torch.nn as nn

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self): 
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/sports-vision/data/combined_dataset"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.num_classes = 1

        # IMPROVEMENT 1: Extended training with better scheduling
        self.max_epoch = 200  # Increased from 100
        self.data_num_workers = 4  # Increased for faster data loading
        self.eval_interval = 5  # Evaluate every 5 epochs to save time
        self.warmup_epochs = 10  # Longer warmup for stability
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.class_names = ["tennis_ball"]
        
        # IMPROVEMENT 2: Multi-scale training for better size handling
        self.multiscale_mode = "range"
        self.img_size = (640, 640)  # Base size
        self.random_size = (14, 26)  # Range: 448px to 832px (32*14 to 32*26)
        self.multiscale_interval = 1  # Change scale every epoch
        
        # IMPROVEMENT 3: Enhanced data augmentation
        self.mosaic_prob = 0.8  # Increased mosaic probability
        self.mixup_prob = 0.3   # Add mixup for better boundary learning
        self.hsv_prob = 0.8     # More color augmentation
        self.flip_prob = 0.5
        

        self.test_size = (640, 640)
        self.test_conf = 0.25   # Lower confidence threshold
        self.nmsthre = 0.45     # Slightly lower NMS threshold
        
        # IMPROVEMENT 5: Loss function weights for better localization
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 50
        self.eval_interval = 5
        
        # IMPROVEMENT 6: Learning rate schedule optimization
        self.no_aug_epochs = 20  # Last 20 epochs without augmentation for fine-tuning
        self.min_lr_ratio = 0.05
        
        # IMPROVEMENT 7: Model architecture tweaks
        self.enable_mixup = True
        self.mixup_epoch = 150  # Use mixup for first 150 epochs
        
    def get_model(self):
        """Override to customize model for tennis ball detection"""
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            
            # IMPROVEMENT 8: Optimized head for single class detection
            head = YOLOXHead(
                self.num_classes, 
                self.width, 
                in_channels=in_channels, 
                act=self.act,
                depthwise=False,
            )
            
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)  # Better initialization for single class
        return self.model
    
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Override to add custom augmentations"""
        from yolox.data import (
            COCODataset,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        
        # IMPROVEMENT 9: Enhanced training transforms
        if not no_aug:
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache_img,
            )
            
            # Apply mosaic and mixup
            dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120,  # Increased for mosaic
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
            )
        else:
            # No augmentation for final epochs
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.test_size,
                preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            )

        # IMPROVEMENT 10: Optimized batch sampler
        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)
        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        
        return DataLoader(dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Evaluation data loader"""
        from yolox.data import COCODataset, ValTransform
        import torch.utils.data
        
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_size": batch_size,
            "sampler": sampler,
            "drop_last": False,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Custom evaluator for tennis ball detection"""
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )