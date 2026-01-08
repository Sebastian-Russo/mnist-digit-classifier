import json
import base64
import boto3
import os
from io import BytesIO
from model_loader import predict

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda handler - receives image, returns prediction
    Same as Django views.py but for Lambda
    """
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))

        # Get base64 image
        if 'image' not in body:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'No image provided'})
            }

        # Decode base64 image
        image_data = base64.b64decode(body['image'])
        image_file = BytesIO(image_data)

        # Make prediction
        prediction, confidence = predict(image_file)

        # Return result
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # CORS
            },
            'body': json.dumps({
                'prediction': int(prediction),
                'confidence': round(float(confidence) * 100, 2)
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }