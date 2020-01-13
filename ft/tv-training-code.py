# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import time

from absl import app, flags
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

flags.DEFINE_string(
    "label_file_path",
    "../data/labels.txt",
    "Path to the file containing the category labels.",
)

flags.DEFINE_string(
    "local_image_dir", "../data/images", "Local directory of the image files to label."
)

flags.DEFINE_string(
    "local_annotation_dir",
    "../data/annotations",
    "Local directory of the image annotations",
)

flags.DEFINE_string(
    "local_manifest_dir",
    "../data/manifests/",
    "Local directory of the annotated image manifests",
)

flags.DEFINE_string(
    "s3_bucket_name", None, "S3 bucket to retrieve images from and upload manifest to."
)

flags.DEFINE_string("s3_image_dir", "data/images/", "Prefix of the s3 image objects.")

flags.DEFINE_string(
    "s3_annotation_dir",
    "data/annotations/",
    "Prefix of the s3 image annotation objects",
)
flags.DEFINE_string(
    "s3_manifest_dir",
    "data/manifests/",
    "Prefix of the s3 image annotation manifest objects",
)


class ODDataSet(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(
    #     in_features_mask, hidden_layer, num_classes
    # )

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(unused_argv):

    start_time = time.time()

    use_s3 = True if flags.FLAGS.s3_bucket_name is not None else False

    if use_s3:
        if not s3_util.s3_bucket_exists(flags.FLAGS.s3_bucket_name):
            use_s3 = False
            print(
                "Bucket: %s either does not exist or you do not have access to it"
                % flags.FLAGS.s3_bucket_name
            )
        else:
            print(
                "Bucket: %s exists and you have access to it"
                % flags.FLAGS.s3_bucket_name
            )

    if use_s3:
        # Download new images from s3
        s3_images = s3_util.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_image_dir,
            flags.FLAGS.image_file_type,
        )
        s3_util.s3_download_files(
            flags.FLAGS.s3_bucket_name, s3_images, flags.FLAGS.local_image_dir
        )

        # Download any nest annotation files from s3
        s3_annotations = s3_util.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_annotation_dir,
            flags.FLAGS.annotation_file_type,
        )

        s3_util.s3_download_files(
            flags.FLAGS.s3_bucket_name,
            s3_annotations,
            flags.FLAGS.local_annotation_dir,
        )

        # Download any new manifests files from s3
        s3_manifests = s3_util.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name, flags.FLAGS.s3_manifest_dir,
        )

        s3_util.s3_download_files(
            flags.FLAGS.s3_bucket_name, s3_manifests, flags.FLAGS.local_manifest_dir
        )

    if not os.path.isfile(flags.FLAGS.label_file_path):
        print("Invalid category labels path.")
        return

    # read in the category labels
    category_labels = open(flags.FLAGS.label_file_path).read().splitlines()

    if len(category_labels) == 0:
        print("No label categories found")
        return

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset("PennFudanPed", get_transform(train=True))
    dataset_test = PennFudanDataset("PennFudanPed", get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    # )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        # num_workers=4,
        num_workers=1,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    app.run(main)
