import boto3
import json

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

def lambda_handler(event, context):
    """
    Proxy Lambda: Browser → API Gateway → Lambda → SageMaker
    """
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))

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
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }