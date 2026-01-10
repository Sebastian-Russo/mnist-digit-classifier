# SageMaker Deployment Notebook Cells

## Cell 1: Setup and Imports
```python
import sagemaker
import boto3
import json
import base64
import numpy as np
from sagemaker.pytorch.model import PyTorchModel

# Get SageMaker execution role
role = sagemaker.get_execution_role()
print(f"Role: {role}")

# SageMaker session
sess = sagemaker.Session()
region = sess.boto_region_name
print(f"Region: {region}")
```

## Cell 2: Update inference.py with Truncated Image Support
```python
# Read current inference.py
with open('inference.py', 'r') as f:
    content = f.read()

# Add the fixes after imports
lines = content.split('\n')
insert_idx = 0
for i, line in enumerate(lines):
    if line.startswith('import ') or line.startswith('from '):
        insert_idx = i + 1
    elif line.strip() == '' and insert_idx > 0:
        break

# Insert the fixes
lines.insert(insert_idx, '')
lines.insert(insert_idx + 1, 'from PIL import ImageFile')
lines.insert(insert_idx + 2, 'ImageFile.LOAD_TRUNCATED_IMAGES = True')
lines.insert(insert_idx + 3, '')
lines.insert(insert_idx + 4, 'import logging')
lines.insert(insert_idx + 5, 'logger = logging.getLogger(__name__)')
lines.insert(insert_idx + 6, 'logger.setLevel(logging.DEBUG)')

# Also update input_fn to add logging
for i, line in enumerate(lines):
    if 'image_data = base64.b64decode(data[\'image\'])' in line:
        lines.insert(i + 1, '    logger.debug(f"Decoded bytes length: {len(image_data)}")')
        lines.insert(i + 2, '    logger.debug(f"First 8 bytes: {image_data[:8].hex()}")')
        break

# Update predict_fn to use BytesIO and add logging
for i, line in enumerate(lines):
    if 'img = Image.open(data).convert(\'L\')' in line:
        lines[i] = '    try:'
        lines.insert(i + 1, '        img = Image.open(data).convert(\'L\')')
        lines.insert(i + 2, '        logger.info(f"Image opened: mode={img.mode}, size={img.size}")')
        lines.insert(i + 3, '    except Exception as e:')
        lines.insert(i + 4, '        logger.exception("Image open failed")')
        lines.insert(i + 5, '        raise')
        break

# Write back
with open('inference.py', 'w') as f:
    f.write('\n'.join(lines))

print("Updated inference.py with truncated image support and logging")
```

## Cell 3: Re-package and Re-deploy Model
```python
# Re-create model.tar.gz
!tar -czf model.tar.gz mnist_model.pth inference.py

# Upload to S3
!aws s3 cp model.tar.gz s3://sagemaker-us-east-1-308665918648/model.tar.gz

# Update the endpoint
from sagemaker.pytorch.model import PyTorchModel

model = PyTorchModel(
    model_data='s3://sagemaker-us-east-1-308665918648/model.tar.gz',
    role=sagemaker.get_execution_role(),
    entry_point='inference.py',
    framework_version='2.1.0',
    py_version='py310'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='mnist-classifier-endpoint',
    update_endpoint=True
)
```

## Cell 4: Test Endpoint with Base64 Padding Fix
```python
import json
import boto3

# Read your test file
with open('test_payload.json', 'r') as f:
    test_payload = json.load(f)

# Fix padding
base64_str = test_payload['image']
padding = len(base64_str) % 4
if padding:
    base64_str += '=' * (4 - padding)
    test_payload['image'] = base64_str

# Test
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='mnist-classifier-endpoint',
    ContentType='application/json',
    Body=json.dumps(test_payload)
)

result = json.loads(response['Body'].read())
print(f"Prediction: {result}")
```

## Cell 5: Move Base64 Padding Fix to inference.py
```python
# Read current inference.py
with open('inference.py', 'r') as f:
    content = f.read()

# Find input_fn and add padding fix
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'image_data = base64.b64decode(data[\'image\'])' in line:
        # Add padding fix before decode
        lines.insert(i, '    # Fix base64 padding automatically')
        lines.insert(i + 1, '    base64_str = data[\'image\']')
        lines.insert(i + 2, '    padding = len(base64_str) % 4')
        lines.insert(i + 3, '    if padding:')
        lines.insert(i + 4, '        base64_str += \'=\' * (4 - padding)')
        lines.insert(i + 5, '        data[\'image\'] = base64_str')
        lines.insert(i + 6, '')
        # Update the original line to use the fixed data
        lines[i + 7] = line.replace('data[\'image\']', 'base64_str')
        break

# Write back
with open('inference.py', 'w') as f:
    f.write('\n'.join(lines))

print("Added automatic base64 padding fix to inference.py")
```

## Cell 6: Re-deploy with Updated inference.py
```python
# Re-create model.tar.gz
!tar -czf model.tar.gz mnist_model.pth inference.py

# Upload to S3
!aws s3 cp model.tar.gz s3://sagemaker-us-east-1-308665918648/model.tar.gz

# Update the endpoint
from sagemaker.pytorch.model import PyTorchModel

model = PyTorchModel(
    model_data='s3://sagemaker-us-east-1-308665918648/model.tar.gz',
    role=sagemaker.get_execution_role(),
    entry_point='inference.py',
    framework_version='2.1.0',
    py_version='py310'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='mnist-classifier-endpoint',
    update_endpoint=True
)
```

## Cell 7: Test Endpoint (No Padding Fix Needed)
```python
import json
import boto3

# Read raw test file (no padding fix needed)
with open('test_payload.json', 'r') as f:
    test_payload = json.load(f)

# Test directly - inference.py handles padding automatically
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='mnist-classifier-endpoint',
    ContentType='application/json',
    Body=json.dumps(test_payload)
)

result = json.loads(response['Body'].read())
print(f"Prediction: {result}")
```

## Notes:
- Cell 1: Basic setup and role retrieval
- Cell 2: Added PIL truncated image support and logging to inference.py
- Cell 3: Re-packaged model with updated inference.py and redeployed endpoint
- Cell 4: Fixed base64 padding issue and tested endpoint (still working on getting successful response)
- Cell 5: Moved base64 padding fix into inference.py (automatic handling)
- Cell 6: Re-deployed with robust inference.py that handles padding automatically
- Cell 7: Test endpoint without needing client-side padding fixes

---

## How Notebook Cells Work

### Basic Concepts
- **Sequential Execution**: Cells run in order. Variables created in Cell 1 are available in all later cells.
- **State Persistence**: Imports, variables, and objects persist across cells within the same kernel session.
- **Independent But Connected**: Each cell can be run individually, but they share the same Python kernel.

### Execution Order
- Cells execute in the order you run them (not necessarily top-to-bottom)
- The `[ ]` shows empty (not run), `[*]` shows running, `[1]` shows it ran 1st
- You can re-run cells to update their state

### Common Pattern Example
```python
# Cell 1: Setup (run once)
import pandas as pd
data = pd.read_csv('file.csv')

# Cell 2: Process (can re-run multiple times)
processed = data.dropna()

# Cell 3: Visualize (re-run to see updates)
processed.plot()
```

### Important Notes
- If you restart the kernel, all cells reset and must be re-run from the top
- Variables persist across cells until kernel restart
- Each cell can be edited and re-run independently
- Use cells to break up long workflows into manageable steps

---

## SageMaker Notebook UI: Files and Running

### Files Tab (Left Panel)
This is your file browser for the notebook instance:
- **`Untitled.ipynb`** - Your active notebook file
- **`inference.py`** - SageMaker inference script
- **`mnist_model.pth`** - Trained PyTorch model
- **`model.tar.gz`** - Packaged model artifact
- **Folders** - Navigate directories
- **Upload** - Upload files from your computer
- **New** - Create new files/folders

### Running Tab (Left Panel)
Shows active processes:
- **`conda_python3`** - The Python kernel/environment
- **`Untitled.ipynb`** - Your running notebook
- **Terminals** - Any open terminal sessions
- **Stop/Restart** - Control running processes

### Key Points
- **Files tab** = File explorer (like Finder/File Explorer)
- **Running tab** = Task manager (like Activity Monitor)
- You can have multiple notebooks open simultaneously
- Each notebook runs in its own kernel
- Stopping a kernel clears all variables from that notebook

### Common Actions
- **Files**: Double-click to open, right-click for options
- **Running**: Click to switch between notebooks, stop kernels that are stuck