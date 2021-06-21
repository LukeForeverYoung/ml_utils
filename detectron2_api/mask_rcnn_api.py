from operator import mod
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.structures import ImageList
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import yaml
import cv2
import torch
from icecream import ic
from detectron2.data import detection_utils
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from PIL import Image

"""
参考
https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/rcnn.py
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py

输入Predictor的是H,W,C BGR 的ndarray 其自动将图像转化成 C,H,W 和模型需要的通道顺序的tensor 组装成[dict]输入model

输入model的是 C,H,W 的list[dict{'image':...}] 模型内会调用 self.preprocess_image 对图像做归一化 to cuda 并用ImageList.from_tensors(images, self.backbone.size_divisibility) 进行组装

下面是RCNN的forward流程
images = self.preprocess_image(batched_inputs)
features = self.backbone(images.tensor)
proposals, _ = self.proposal_generator(images, features, None)
detected_instances = [x.to(self.device) for x in detected_instances]
"""


def make_cfg(config_path):
    # Create config
    cfg = get_cfg()
    # cfg.merge_from_file("mask_rcnn/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.merge_from_file(config_path)
    return cfg


def get_model(cfg):
    model = build_model(cfg)  # returns a torch.nn.Module
    if cfg.MODEL.WEIGHTS:
        DetectionCheckpointer(model).load(
            cfg.MODEL.WEIGHTS
        )  # load a file, usually from cfg.MODEL.WEIGHTS
    else:
        print("no weight loaded.")
    return model


class MRCNN:
    def __init__(self, config_path):
        self.cfg = make_cfg(config_path)
        self.model = get_model(self.cfg)

        import detectron2.data.transforms as T

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )

    def predict(self, batched_inputs: list, training=False):
        if training:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)
        images = self.model.preprocess_image(batched_inputs)
        # ['p2', 'p3', 'p4', 'p5', 'p6']
        # Image feature
        features = self.model.backbone(images.tensor)

        # [Instances(),] 每张图一个Instances实例
        proposals, _ = self.model.proposal_generator(images, features)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        instances = self.model._postprocess(results, batched_inputs, images.image_sizes)
        instances = [_["instances"] for _ in instances]

        # region features
        mask_features = [features[f] for f in self.model.roi_heads.in_features]
        mask_features = self.model.roi_heads.mask_pooler(
            mask_features, [x.pred_boxes for x in instances]
        )
        ic(instances)
        res = {
            "features": features,
            "instances": instances,
            "mask_features": mask_features,
        }

        self.visualize(batched_inputs, instances)
        # 恢复训练状态
        self.model.train()
        torch.set_grad_enabled(True)
        return res

    def extract_feature_pipeline(self, files):
        batched_inputs = [self.read_image(_) for _ in files]
        res = self.predict(batched_inputs)
        return res

    def read_image(self, img_path):
        # 以BGR读入,可以用cv2.read替换
        # original_image=detection_utils.read_image("./example/input.jpg",format="BGR") #(H,W,C)
        original_image = cv2.imread(img_path)
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        ic(inputs)
        return inputs

    def visualize(self, batched_inputs, instances):
        for inp, ins in zip(batched_inputs, instances):
            img = inp["image"]
            img = detection_utils.convert_image_to_rgb(
                img.permute(1, 2, 0), self.model.input_format
            )
            img = Image.fromarray(img.astype("uint8"), "RGB")
            img = img.resize((batched_inputs[0]["width"], batched_inputs[0]["height"]))
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(boxes=ins.pred_boxes.tensor.cpu().numpy())
            prop_img = v_pred.get_image()
            plt.imshow(prop_img)
            plt.show()


if __name__ == "__main__":
    mrcnn = MRCNN("pretrain_models/mask_rcnn/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    mrcnn.extract_feature_pipeline(["example/input.jpg"])

    """

    cfg = get_cfg()
    cfg.merge_from_file("mask_rcnn/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Make prediction
    im=cv2.imread("./example/input.jpg")
    ic(im.shape)
    outputs = predictor(im)
    insts=outputs['instances']
    print(insts)
    """
