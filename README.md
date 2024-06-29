# Harvested-Arecanut-Variety-Classification-using-ML
Sure! Below is a detailed README for your GitHub repository, incorporating the information from your introduction and abstract. You can modify it as needed to better fit your project's specifics and preferences.

---

# Automated Arecanut Classification and Detection

## Introduction

The collection of harvested arecanuts from the ground is a fundamental step in the processing chain, but it often presents significant challenges, particularly in sorting them by variety. In many areca farms, nuts can fall into drainage systems, posing a hazard to laborers who must navigate these areas to collect them, risking skin allergies and other health issues. Additionally, hand-picking of nuts requires a considerable amount of labor, which has become increasingly expensive due to the decline in manual labor availability and the time-consuming nature of the process.

The ability to classify arecanuts based on their characteristics such as size, color, and quality is crucial for commercialization, as it directly affects pricing and market demand. To address these challenges, there is a need for an automated system capable of accurately identifying and collecting arecanuts based on their variety. Such a system would not only improve precision and consistency but also enhance efficiency in the grading process.

## Abstract

In this study, we designed and developed a robust classification and detection mechanism using machine learning algorithms to distinguish between four distinct classes of harvested arecanuts:
1. Isolated Ripened Yellow
2. Clustered Ripened Yellow
3. Isolated UnRipened Green
4. Clustered UnRipened Green ArecaNuts

We implemented multiple classification algorithms, including Support Vector Machines (SVM), Random Forest, K-Nearest Neighbours (KNN), and Neural Networks, for categorizing arecanuts. Additionally, we trained a YOLOv8 model for object detection to accurately identify and classify arecanuts in various scenarios.

The results demonstrate that our machine learning-based approach significantly reduces manual labor and operational costs while enhancing the accuracy and efficiency of arecanut classification. This research highlights the potential of advanced machine learning techniques to address critical challenges in the agricultural sector, paving the way for more efficient and automated harvesting solutions.

## Project Structure

- `data/`: Directory containing the dataset used for training and validation.
- `notebooks/`: Jupyter notebooks for data exploration, model training, and evaluation.
- `src/`: Source code for model training, inference, and utility functions.
- `models/`: Pre-trained models and saved weights.
- `results/`: Directory to save the results of the model predictions.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/arecanut-classification.git
    cd arecanut-classification
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up YOLOv8:**
    Follow the [Ultralytics YOLOv8 installation guide](https://github.com/ultralytics/yolov5) to install and set up YOLOv8.

4. **Download the dataset:**
    Download the dataset from [Roboflow](https://universe.roboflow.com/husked-arecanut-variety-detection/harvested-arecanut-detection/dataset/1) and place it in the `data/` directory.

## Results

The proposed system demonstrates the potential to significantly reduce manual labor and operational costs, paving the way for more efficient arecanut harvesting and processing practices.

## Contributing

If you would like to contribute to this project, please open an issue or submit a pull request. We welcome all contributions that improve the functionality or performance of the system.

## License

This project is licensed under the MIT License.

## Acknowledgements

We would like to thank the [Roboflow](https://universe.roboflow.com) community for providing the tool to annotate the dataset and resources that made this project possible.

