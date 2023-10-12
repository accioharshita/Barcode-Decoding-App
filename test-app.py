import torch
import cv2
from pyzbar import pyzbar 
from flask import Flask, request, Response, render_template
import numpy as np


app= Flask(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)


@app.route('/decode', methods= ['GET', 'POST'])
def decode():
    if request.method == 'POST':

        # Handle the uploaded image here
        image = request.files['image'].read()

        #load the input image
        nparr = np.fromstring(image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        results = model(frame)
        detections = results.pandas().xyxy[0]

        # Extract the coordinates, class id, and confidence for each detection
        for i, detection in detections.iterrows():
            x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
            x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]

            class_id = detection['class']
            confidence = detection['confidence']
            print(f'Detection {i}: class {class_id}, confidence {confidence}, bbox [{x1}, {y1}, {x2}, {y2}]')

            # Draw bounding box on input image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label text on bounding box
            label = f'{"Barcode"} {confidence:.2f}'
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - baseline), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Crop image to bounding box of first detected object
            cropped_img = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(gray)
            for barcode in barcodes:
                data = barcode.data.decode("utf-8")

                # Put barcode data on bounding box
                data_size, baseline = cv2.getTextSize(data, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y2), (x1 + data_size[0], y2 + data_size[1]), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, data, (x1, y2 + data_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Encode the annotated image as JPEG and return it
            _, encoded_image = cv2.imencode('.jpg', frame)
            response = Response(response=encoded_image.tobytes(), content_type='image/jpeg')
            return response


    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)