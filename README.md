# BETTER - Backend API

**BETTER** is a comprehensive FastAPI-based backend system for neural model analysis, explainable AI, and adversarial robustness testing. The system provides advanced capabilities for hierarchical clustering analysis, adversarial attack detection, and model interpretability.

## Features

- **Near Misses Analysis (NMA)**: Hierarchical clustering and visualization of neural network predictions
- **Adversarial Attack Detection**: Generate, detect, and analyze adversarial examples using multiple attack methods
- **Explainable AI**: Query-based model explanations and verbal descriptions of predictions
- **Whitebox Testing**: Analyze model behavior across different label categories
- **Dendrogram Visualization**: Interactive hierarchical clustering of model predictions
- **Multi-Dataset Support**: CIFAR-100, ImageNet, and custom datasets
- **Cloud Storage Integration**: AWS S3 integration for scalable model and data storage

## Architecture

### Tech Stack
- **Framework**: FastAPI with async support
- **ML Libraries**: TensorFlow/Keras, scikit-learn, NumPy
- **Data Processing**: Pandas, matplotlib, PIL
- **Cloud Storage**: AWS S3 with boto3
- **Authentication**: JWT-based authentication
- **Testing**: pytest with comprehensive test suites

### Project Structure
```
BETTER/
├── app.py                     # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── pyrightconfig.json        # Type checking configuration
├── routers/                  # API route handlers
│   ├── adversarial_router.py # Adversarial attack endpoints
│   ├── dendrogram_router.py  # Hierarchical clustering endpoints
│   ├── nma_router.py         # Neural model analysis endpoints
│   ├── query_router.py       # Model explanation endpoints
│   ├── users_router.py       # User management endpoints
│   └── ...
├── services/                 # Business logic layer
│   ├── adversarial_attacks_service.py
│   ├── models_service.py
│   ├── dataset_service.py
│   └── ...
├── request_models/          # Pydantic request/response models
├── utilss/                  # Utility modules
│   ├── classes/            # Core domain classes
│   ├── enums/              # Enumeration definitions
│   └── s3_connector/       # AWS S3 integration
├── tests/                  # Comprehensive test suites
└── data/                  # Dataset information and storage
```

## Setup and Installation

### Prerequisites
- Python 3.11
- AWS Account (for S3 storage)
- Virtual environment support

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BETTER
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file with the following variables:
   ```env
   S3_USERS_BUCKET_NAME=your-s3-bucket-name
   S3_DATASETS_BUCKET_NAME=your-datasets-bucket
   PATH=your-local-path
   MODELS_PATH=your-models-path
   AWS_ACCESS_KEY_ID=your-aws-access-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret-key
   JWT_SECRET_KEY=your-jwt-secret
   ```

5. **Run the server**
   ```bash
   uvicorn app:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`

## Data Storage Structure

The system uses a hierarchical folder structure for organizing user data, models, and analysis results:

```
users/
├── users.json                 # User registry
└── {user_id}/
    ├── models.json            # User's model registry
    ├── current_model.json     # Active model configuration
    └── {model_id}/
        ├── similarity/        # Similarity-based analysis
        │   ├── dendrogram.json
        │   ├── edges_df.csv
        │   └── logistic_regression_model.json
        ├── dissimilarity/     # Dissimilarity-based analysis
        │   └── ...
        └── count/             # Count-based analysis
            └── ...
```
## API Documentation

### Authentication
All endpoints require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

### Core Endpoints

#### Neural Model Analysis (NMA)
Analyze neural networks using hierarchical clustering techniques.

- **Endpoint**: `POST /api/nma/{model_id}`
- **Description**: Perform neural model analysis with hierarchical clustering
- **Request Body**:
  ```json
  {
    "dataset": "imagenet|cifar100",
    "graph_type": "similarity|dissimilarity|count",
    "min_confidence": 0.8,
    "top_k": 4,
    "model_filename": "your_model.keras"
  }
  ```

#### Dataset Labels
Get available labels for supported datasets.

- **Endpoint**: `GET /api/datasets/{dataset_name}/labels`
- **Supported datasets**: `imagenet`, `cifar100`
- **Response**: Array of available class labels

#### Dendrogram Analysis
Generate and manipulate hierarchical clustering dendrograms.

- **Endpoint**: `POST /api/dendrograms`
- **Request Body**:
  ```json
  {
    "model_id": "uuid",
    "graph_type": "count|similarity|dissimilarity",
    "selected_labels": ["cat", "dog", "bird"]
  }
  ```

#### Cluster Naming
Update cluster names in dendrograms for better interpretability.

- **Endpoint**: `PUT /api/dendrograms/auto_naming`
- **Request Body**:
  ```json
  {
    "model_id": "uuid",
    "graph_type": "count",
    "selected_labels": ["Persian_cat", "tabby"],
    "cluster_id": 1263,
    "new_name": "Cats"
  }
  ```

#### Query Analysis
Get detailed explanations for model predictions.

- **Endpoint**: `POST /api/query`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `current_model_id`: UUID of the model
  - `graph_type`: Analysis type
  - `image`: Image file for analysis

#### Whitebox Testing
Analyze model behavior across different label categories.

- **Endpoint**: `POST /api/whitebox_testing`
- **Request Body**:
  ```json
  {
    "model_id": "uuid",
    "graph_type": "count",
    "source_labels": ["Persian_cat", "tabby"],
    "target_labels": ["pug", "boxer"]
  }
  ```

### Adversarial Attack Detection

#### Generate Adversarial Detector
Create a detector model for identifying adversarial examples.

- **Endpoint**: `POST /api/adversarial/generate`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `current_model_id`: UUID of the target model
  - `graph_type`: Analysis type
  - `clean_images` (optional): Clean example images (.npy files)
  - `adversarial_images` (optional): Adversarial example images (.npy files)

**Example with images**:
```bash
curl -X POST "http://127.0.0.1:8000/api/adversarial/generate" \
  -F "current_model_id=uuid" \
  -F "graph_type=similarity" \
  -F "clean_images=@img1.npy" \
  -F "adversarial_images=@adv1.npy"
```

#### Detect Adversarial Examples
Classify images as clean or adversarial using trained detectors.

- **Endpoint**: `POST /api/adversarial/detect`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `current_model_id`: UUID of the model
  - `graph_type`: Analysis type
  - `image`: Image file to analyze
  - `detector_filename`: Detector model filename

**Example**:
```bash
curl -X POST "http://127.0.0.1:8000/api/adversarial/detect" \
  -F "current_model_id=uuid" \
  -F "graph_type=similarity" \
  -F "image=@test_image.jpg" \
  -F "detector_filename=detector.pkl"
```

#### Adversarial Attack Analysis
Generate and analyze adversarial examples using various attack methods.

- **Endpoint**: `POST /api/adversarial/analyze`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `current_model_id`: UUID of the model
  - `graph_type`: Analysis type
  - `image`: Original image file
  - `attack_type`: Attack method (`pgd`, `fgsm`, `deepfool`)
  - `detector_filename`: Detector model filename
  - `epsilon` (optional): Attack strength parameter
  - `alpha` (optional): Step size for iterative attacks
  - `num_steps` (optional): Number of attack iterations
  - `overshoot` (optional): DeepFool overshoot parameter

**Example**:
```bash
curl -X POST "http://127.0.0.1:8000/api/adversarial/analyze" \
  -F "current_model_id=uuid" \
  -F "graph_type=similarity" \
  -F "image=@original_image.jpg" \
  -F "attack_type=pgd" \
  -F "detector_filename=detector.pkl" \
  -F "epsilon=0.03"
```

### Response Formats

#### Standard Success Response
```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully"
}
```

#### Error Response
```json
{
  "detail": "Error description",
  "status_code": 400
}
```

#### Analysis Result Response
```json
{
  "original_image": "base64_encoded_image",
  "original_predictions": [["class_name", 0.95]],
  "original_verbal_explanation": ["explanation_text"],
  "adversarial_image": "base64_encoded_image",
  "adversarial_predictions": [["class_name", 0.85]],
  "detection_result": "Clean|Adversarial",
  "probability": 0.92
}
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/adversarial_tests/
pytest tests/dendrogram_tests/
pytest tests/query_tests/

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Service Tests**: Business logic validation
- **S3 Tests**: Cloud storage integration testing

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `S3_USERS_BUCKET_NAME` | S3 bucket for user data | Yes |
| `S3_DATASETS_BUCKET_NAME` | S3 bucket for datasets | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Yes |
| `JWT_SECRET_KEY` | JWT signing secret | Yes |
| `PATH` | Local storage path | No |
| `MODELS_PATH` | Model storage path | No |

### Supported Attack Methods
- **FGSM** (Fast Gradient Sign Method)
- **PGD** (Projected Gradient Descent)
- **DeepFool** (Geometric adversarial attacks)

### Supported Datasets
- **CIFAR-100**: 100-class image classification
- **ImageNet**: Large-scale image recognition

## Performance and Scaling

### Optimization Features
- **Async Processing**: Non-blocking API operations
- **S3 Integration**: Scalable cloud storage
- **Batch Processing**: Efficient multi-image analysis
- **Caching**: Preprocessing function caching
- **Memory Management**: Optimized for large models

### Resource Requirements
- **Memory**: 8GB+ RAM recommended
- **Storage**: Variable based on model sizes
- **Network**: Stable internet for S3 operations

## Error Handling

The API implements comprehensive error handling:

- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation errors
- **500 Internal Server Error**: Server-side errors

All errors include detailed messages and suggested solutions.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Use type hints throughout the codebase

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation for common solutions
- Review the test files for usage examples

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- FastAPI team for the excellent web framework
- Contributors to the open-source libraries used in this project
