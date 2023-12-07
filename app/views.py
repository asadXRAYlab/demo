import tempfile
import cv2
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import numpy as np
from ultralytics import YOLO
import os
# Create your views here.
def yolo_detect(image_bytes):
    try:
        # Load the YOLO model from the specified ONNX file
        model = YOLO('yolov8l.pt')

        image_array = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        result = model(original_image)
        res_plotted = result[0].plot()

        return res_plotted

    except Exception as e:
        return {'error': str(e)}


def yolo_segment(image_bytes):
    try:
        # Load the YOLO model from the specified ONNX file
        model = YOLO('yolov8l-seg.pt')

        image_array = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        result = model(original_image)
        res_plotted = result[0].plot()

        return res_plotted

    except Exception as e:
        return {'error': str(e)}
    
@csrf_exempt
def yolo_api(request):
    if request.method == 'POST':
        if 'image' in request.FILES and 'choice' in request.POST:
            image = request.FILES['image'].read()

            choice = request.POST.get('choice', None)


            if choice == 'detection':
                detections = yolo_detect(image)
            elif choice == 'segmentation':
                detections = yolo_segment(image)
            else:
                return HttpResponse("Invalid choice")

                # Save the annotated image to a temporary file
            temp_output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
            cv2.imwrite(temp_output_path, detections)

                # Return the annotated image as a response
            with open(temp_output_path, 'rb') as image_file:
                response = HttpResponse(image_file.read(), content_type='image/jpeg')
                response['Content-Disposition'] = 'inline; filename=output.jpg'
                return response
            
        else:
            return JsonResponse({'error': 'Image or model_path not provided'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)