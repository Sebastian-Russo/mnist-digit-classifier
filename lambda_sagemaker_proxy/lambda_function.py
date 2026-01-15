import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

def lambda_handler(event, context):
    """
    Proxy Lambda: Browser → API Gateway → Lambda → SageMaker
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        # FIX: API Gateway sends data directly, not wrapped in 'body'
        # Check if event has 'body' key (HTTP API format) or is direct JSON (REST API with mapping template)
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event

        logger.info(f"Parsed body: {body}")

        if 'image' not in body:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'No image provided'})
            }

        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='mnist-endpoint-sebastian-v4',
            ContentType='application/json',
            Body=json.dumps(body)
        )

        # Parse SageMaker response
        result = json.loads(response['Body'].read())

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }