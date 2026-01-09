import os
import argparse
import cv2
import torch
import numpy as np
import supervision as sv

# GroundingDINO and SAM imports
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

def get_args():
    parser = argparse.ArgumentParser(description="AV-Barrier Auto-Labeling Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="jersey barrier . traffic cone . guardrail", help="Text prompt for DINO")
    parser.add_argument("--output", type=str, default="./outputs", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

class AutoLabeler:
    def __init__(self, device):
        self.device = device
        
        # Paths - Ensure these match your weights folder
        DINO_CONFIG = "core/GroundingDINO_SwinT_OGC.py"
        DINO_CHECKPOINT = "weights/groundingdino_swint_ogc.pth"
        SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
        
        print(f"🚀 Loading models onto {self.device}...")
        self.dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT, device=self.device)
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def process_image(self, image_path, text_prompt):
        # 1. Detection (GroundingDINO)
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=0.35,
            text_threshold=0.25,
            device=self.device
        )

        if len(boxes) == 0:
            print(f"⚠️ No objects found for prompt: '{text_prompt}'")
            return image_source, None

        # 2. Segmentation (SAM)
        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]).to(self.device)
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # 3. Create Supervision Detections
        detections = sv.Detections(
            xyxy=boxes_xyxy.cpu().numpy(),
            confidence=logits.cpu().numpy(),
            class_id=np.zeros(len(boxes_xyxy), dtype=int),
            mask=masks.cpu().numpy().squeeze(1)
        )
        return image_source, detections

def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)

    labeler = AutoLabeler(device=args.device)
    image_source, detections = labeler.process_image(args.image, args.prompt)

    if detections is not None:
        # Annotate
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        
        annotated_frame = mask_annotator.annotate(scene=image_source.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

        # Save Visual Result
        save_path = os.path.join(args.output, "annotated_" + os.path.basename(args.image))
        cv2.imwrite(save_path, annotated_frame)
        print(f"✅ Visualization saved to {save_path}")

        # Export Labels (YOLO format)
        dataset = sv.DetectionDataset(
            classes=args.prompt.split(" . "),
            images={args.image: image_source},
            annotations={args.image: detections}
        )
        dataset.as_yolo(
            images_directory_path=os.path.join(args.output, "images"),
            annotations_directory_path=os.path.join(args.output, "labels")
        )
        print(f"📦 YOLO labels exported to {args.output}/labels")

if __name__ == "__main__":
    main()