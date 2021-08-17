# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="../configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD33.yaml",# test_e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes_car
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=['MODEL.WEIGHT','./','MODEL.DOMAIN_ADAPTATION_ON',False,],#'DATASETS.TEST',('visual_cocostyle',)
        nargs=argparse.REMAINDER,type=str
    )
    basedir='../'
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    filename=os.path.splitext(os.path.basename(args.config_file))[0]
    label_flag=filename[-2]
    unlabel_flag=filename[-1]
    model_dir = basedir+"log%s%s/"%(label_flag,unlabel_flag)
    save_dir=os.path.join(model_dir,'test')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    last_checkpoint=os.path.join(model_dir,'last_checkpoint')
    if os.path.exists(last_checkpoint):
        with open(last_checkpoint,'r') as f:
            lines=f.readlines()
            cfg.MODEL.WEIGHT=os.path.join(model_dir,os.path.basename(lines[-1]))
    else:
        cfg.MODEL.WEIGHT=os.path.join(model_dir,'model_final.pth')

    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR,model_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        if os.path.exists(os.path.join(output_folder, 'predictions.pth')):
            predictions=torch.load(os.path.join(output_folder, 'predictions.pth'))
            extra_args = dict(
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                iou_types=iou_types,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            )

            evaluate(dataset = data_loader_val.dataset,
                     predictions=predictions,
                     output_folder=output_folder,
                     **extra_args)

        else:
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
        synchronize()


if __name__ == "__main__":
    main()
