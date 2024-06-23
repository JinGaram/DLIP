# -------------------------------------------------------------------------------------------------
# * @author  21900727 Garam Jin & 22100034 Eunji Ko
# * @Date    2024-06-24
# * @Mod	 2024-06-10 by YKKIM
# * @brief   Final Project(DLIP)
# -------------------------------------------------------------------------------------------------

from ultralytics import YOLO

def train():
    # Load a pretrained YOLO model
    model = YOLO('yolov8n-seg.pt')

    # Train the model using the 'maskdataset.yaml' dataset for 3 epochs
    results = model.train(data='Yolo_segmentation.yaml', epochs=12, task='segment')
    
if __name__ == '__main__':  
    train()
