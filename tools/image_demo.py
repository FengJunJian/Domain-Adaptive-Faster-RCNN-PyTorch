# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import json
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import evaluate_predictions_on_coco, \
    prepare_for_coco_detection
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data.datasets.evaluation import evaluate

from ship_predictor import COCODemo

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import os


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD30.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=600,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        # default=True,
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=['OUTPUT_DIR', 'log30'],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    Video_flag = False
    if Video_flag:
        cam = cv2.VideoCapture(0)
        while True:
            start_time = time.time()
            ret_val, img = cam.read()
            composite = coco_demo.run_on_opencv_image(img)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            cv2.imshow("COCO detections", composite)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
    else:
        img_root = 'E:/fjj/SeaShips_SMD/JPEGImages'
        img = cv2.imread(os.path.join(img_root, 'MVI_1474_VIS_00120.jpg'))
        start_time = time.time()
        composite, predictions = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))

        cv2.imshow("COCO detections", composite)

        cv2.waitKey()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
