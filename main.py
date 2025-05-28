from ultralytics import YOLO
import os
import time


class AuditModel:
    def __init__(self, train_data: str, test_images: str, test_model: str = "runs/segment/train/weights/best.pt") -> None:
        self.train_data = train_data
        self.test_images = test_images
        self.test_model = test_model

    def train(self):
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=f'{self.train_data}/data.yaml',   # Path to dataset configuration file
            epochs=50,                             # Number of training epochs
            imgsz=640,                             # Image size for training
            plots=True,                            # Enable training plots
            patience=15,                           # Early stopping patience (epochs with no improvement)
            warmup_epochs=5,                       # Number of warmup epochs for learning rate
            lr0=0.01,                              # Initial learning rate
            lrf=0.01,                              # Final learning rate (as a fraction of initial)
            cos_lr=True,                           # Use cosine learning rate decay
            label_smoothing=0.2,                   # Label smoothing factor (helps reduce overfitting)
            translate=0.08,                        # Image translation augmentation (fraction)
            scale=0.1,                             # Image scaling augmentation (fraction)
            shear=0.0,                             # Image shear augmentation (fraction)
            perspective=0.001,                     # Perspective augmentation (fraction)
            mosaic=0.5,                            # Mosaic augmentation probability (lower to reduce artifacts)
            mixup=0.0,                             # MixUp augmentation probability (disabled for higher accuracy)
            copy_paste=0.0,                        # Copy-Paste augmentation probability (disabled)
            fliplr=0.5,                            # Probability of horizontal flip
            flipud=0.0,                            # Probability of vertical flip
            box=0.05,                              # Box loss gain (weight for bounding box loss)
            cls=0.6,                               # Class loss gain (weight for classification loss)
            dfl=1.0                                # Distribution Focal Loss gain
        )
    
    def val(self):
        model = YOLO(self.test_model)
        metrics = model.val(data=f'{self.train_data}/data.yaml')
        print(metrics)
        print(model.model.task)
    
    def predict(self):
        model = YOLO(self.test_model)
        start = time.time()

        for img_name in os.listdir(self.test_images):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.test_images, img_name)
                results = model.predict(img_path, task="segment", save=True, conf=0.1, imgsz=418)

                for result in results:
                    if result:
                        classes = result.boxes.cls
                        confs = result.boxes.conf
                        masks = result.masks.data
                        for cls, conf in zip(classes, confs):
                                print(f"{model.names[int(cls)]} ({float(conf):.2f})")

        print(f"Processadas: {len(os.listdir(self.test_images))} imagems - Tempo total: {time.time() - start:.2f} segundos")


if __name__ == "__main__":
    train_data = "cable_test"
    test_images = "test_data"

    audit_model = AuditModel(train_data, test_images, "runs/segment/train2/weights/best.pt")

    # audit_model.train()
    # audit_model.val()
    audit_model.predict()