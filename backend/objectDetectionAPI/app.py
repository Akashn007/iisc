from PIL import Image
import io
import pandas as pd
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

#import yolov8
model = YOLO("yolov8n.pt")
model.fuse()

def get_image_from_bytes(binary_image):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image

def get_bytes_from_image(image):

    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  
    return_image.seek(0)  
    return return_image


def transform_predict_to_df(results, labeles_dict):
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model, input_image, save = False, image_size = 1248, conf = 0.5, augment = False):
    
    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save, 
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        )
    
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


################################# BBOX Func #####################################

def add_bboxs_on_img(image, predict):
   
    annotator = Annotator(np.array(image))
    predict = predict.sort_values(by=['xmin'], ascending=True)

    for i, row in predict.iterrows():

        text = f"{row['name']}: {int(row['confidence']*100)}%"
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        annotator.box_label(bbox, text, color=colors(row['class'], True))
    return Image.fromarray(annotator.result())


################################# Models #####################################


def detect_sample_model(input_image):
    predict = get_model_predict(
        model=model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict