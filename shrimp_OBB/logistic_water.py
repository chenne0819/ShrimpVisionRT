from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the model class
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# Prediction function
def predict_image_class(image):
    """
    Predict the class of a single image.

    Parameters:
    - image (numpy.ndarray or str): Input image. Can be an OpenCV-style numpy.ndarray or a file path.

    Returns:
    - predicted_class (str): The predicted class label.
    - confidence (float): Prediction confidence score.
    """

    # If input is a file path, open with PIL; if it's numpy.ndarray, convert to PIL format
    if isinstance(image, str):
        image = Image.open(image).convert('L')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('L')
    else:
        raise ValueError("Input should be a file path (str) or a numpy.ndarray!")

    # Model parameters
    input_dim = 28 * 28
    num_classes = 2
    class_names = ['turbid', 'clear']
    model_path = './Model/logistic_regression_model.pth'  # Path to the trained model

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the model
    model = LogisticRegressionModel(input_dim=input_dim, output_dim=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Preprocess the input image
    image = transform(image).view(-1, input_dim)

    # Run prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

    return predicted_class, confidence
