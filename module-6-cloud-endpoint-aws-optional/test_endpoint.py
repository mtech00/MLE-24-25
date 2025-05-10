import boto3
import json

# Create SageMaker runtime client
client = boto3.client('sagemaker-runtime', region_name='eu-north-1')  # Change region if needed

# Sample data - modify with features appropriate for your model
payload = json.dumps({"features": [[5.1, 3.5, 1.4, 0.2]]})

# Call endpoint
response = client.invoke_endpoint(
    EndpointName='my-model-endpoint',
    ContentType='application/json',
    Body=payload
)

# Parse response
result = json.loads(response['Body'].read().decode())
print("Endpoint response:")
print(json.dumps(result, indent=2))
