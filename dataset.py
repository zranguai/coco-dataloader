# 自定义数据集
import cv2
import torch

from lib import Dataset
from pycocotools.coco import COCO
import os
import numpy as np


# # 简单测试
# class SimpleV1Dataset(Dataset):
#     def __init__(self):
#         # 伪造数据
#         self.imgs = np.arange(0, 16).reshape(8, 2)
#
#     def __getitem__(self, index):
#         return self.imgs[index]
#
#     def __len__(self):
#         return self.imgs.shape[0]


class CocoDataset(Dataset):
    def __init__(self,
                 img_path="/home/zranguai/Python-Code/Only-test/coco/val2017",
                 ann_path="/home/zranguai/Python-Code/Only-test/coco/annotations/instances_val2017.json",
                 mode="train"):
        assert mode in ["train", "val", "test"]
        self.img_path = img_path
        self.ann_path = ann_path
        self.mode = mode

        self.data_info = self.get_data_info()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        if self.mode == "train":
            data = self.get_train_data(index)
        return data

    def get_data_info(self):
        self.coco_api = COCO(self.ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())  # 获得coco类别的id
        self.cat2labels = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)  # 得到categories里面信息
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)  # json文件里面的images信息
        return img_info

    def get_train_data(self, idx):
        img_info = self.get_per_img_info(idx)
        file_name = img_info["file_name"]
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)  # 图片信息

        ann = self.get_img_annotation(idx)  # 读取标签(bbox, label)信息
        meta = dict(
            img=img, img_info=img_info, gt_bboxes=ann["bboxes"], gt_labels=ann["labels"]
        )

        # 数据增强操作(可选)
        # meta = self.pipeline(meta, self.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))  # img转换成(C,H,W)然后转换成torch格式
        return meta

    def get_per_img_info(self, idx):
        """ 该函数主要为了检测id是否是int类型
        得到每张图片的详细信息
        { 'date_captured': '2021',
          'file_name': '000000000005.jpg',
          'id': 5,
          'height': 640,
          'width': 481}
        :param idx:
        :return:
        """
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]  # id应该是int类型
        if not isinstance(id, int):
            raise TypeError("Image id must be int")
        info = {"file_name": file_name, "height": height, "width": width, "id": id}
        return info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])  # 根据img_id得到ann_ids(例如imgid:1 ann_ids:[1, 2, 3, 4, 5, 6, 7, 8])
        anns = self.coco_api.loadAnns(ann_ids)  # 根据ann_ids得到所有的关于这个img的标签信息

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for ann in anns:
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]  # xyxy
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)  # 坐标
                gt_labels.append(self.cat2labels[ann["category_id"]])  # label

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bbox_ignore=gt_bboxes_ignore
        )
        return annotation


