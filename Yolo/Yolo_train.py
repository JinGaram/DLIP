from ultralytics import YOLO

def train():
    # Load a pretrained YOLO model
    model = YOLO('yolov8n-seg.pt')

    # Train the model using the 'maskdataset.yaml' dataset for 3 epochs
    results = model.train(data='Yolo_segmentation.yaml', epochs=12, task='segment')
    
if __name__ == '__main__':  
    train()