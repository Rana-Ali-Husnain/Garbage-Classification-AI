# Garbage Classification AI

An AI-powered image classification application that identifies waste categories from an uploaded image. The project uses MobileNetV2 transfer learning and a Streamlit interface.

## 📊 Dataset

**Dataset Source:** Kaggle

🔗 **[Click Here to View and Access the Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)**

The dataset contains **15,515 images across 12 garbage categories** and is used for training and evaluating the model.

## Project Overview

The application predicts one of 12 categories:

-   Battery
-   Biological
-   Brown Glass
-   Cardboard
-   Clothes
-   Green Glass
-   Metal
-   Paper
-   Plastic
-   Shoes
-   Trash
-   White Glass

Users upload an image and receive the predicted category, confidence
score, and top three predictions.

## Model and Results

-   **Architecture:** MobileNetV2 (ImageNet pretrained, transfer
    learning)
-   **Input size:** 224 × 224 × 3
-   **Number of classes:** 12
-   **Test images:** 3,103
-   **Test accuracy:** 94%
-   **Macro F1-score:** 0.92
-   **Weighted F1-score:** 0.94

These results are from the held-out test split used in this project.
Performance may vary on images from different sources and real-world
conditions.

### Per-class F1-score

  Category        F1-score
  ------------- ----------
  Battery             0.95
  Biological          0.97
  Brown Glass         0.92
  Cardboard           0.93
  Clothes             0.98
  Green Glass         0.92
  Metal               0.84
  Paper               0.92
  Plastic             0.82
  Shoes               0.96
  Trash               0.92
  White Glass         0.85

## Project Structure

``` text
Garbage-Classification-AI/
├── model/
│   └── best_garbage_model.keras
├── app.py
├── requirements.txt
└── README.md
```

> The current app code loads `model/best_garbage_model.keras`. If you
> switch to separate JSON architecture and weights files, update the
> model-loading code and this structure accordingly.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR-USERNAME/Garbage-Classification-AI.git
cd Garbage-Classification-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment (recommended)

On Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Start the app

``` bash
python -m streamlit run app.py
```

The app usually opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1.  Push the project to a GitHub repository.
2.  Open [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Sign in with GitHub and authorize repository access.
4.  Select **Create app**.
5.  Choose the repository, branch, and `app.py` as the main file.
6.  Deploy.

Make sure the model file exists at the path expected by `app.py`. Check
GitHub file-size limits before committing a large model. If it is too
large, host it separately and update the app to load it from that host.

## Dataset

The project uses a garbage image dataset containing 15,515 images across
12 classes. The class distribution is imbalanced, with Clothes
containing substantially more images than several other classes.

Add the exact Kaggle dataset URL and its license/attribution here before
publishing.

## Technologies Used

-   Python
-   TensorFlow / Keras
-   MobileNetV2
-   NumPy
-   Pillow
-   Streamlit
-   Scikit-learn

## Future Improvements

-   Improve performance for Plastic, Metal, and White Glass.
-   Add Grad-CAM visual explanations.
-   Test with more real-world images, backgrounds, and lighting
    conditions.
-   Add recycling and disposal guidance for each predicted category.
-   Compare additional model architectures and training strategies.

## Disclaimer

This is an educational AI prototype. Predictions are model estimates and
should not be treated as definitive waste-management or recycling
advice.

## Project Resources

- **Live Application:** [View Live App](YOUR_LIVE_APP_LINK)
- **Project Report:** [View Project Report](YOUR_REPORT_LINK)
- **YouTube Demonstration:** [Watch Demo](YOUR_YOUTUBE_LINK)
- **LinkedIn:** [Connect on LinkedIn](YOUR_LINKEDIN_LINK)
- **Project Presentation:** [View PPT](YOUR_PPT_LINK)
- **GitHub Repository:** [View Source Code](https://github.com/Rana-Ali-Husnain/Garbage-Classification-AI)
- **Dataset:** [View Kaggle Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
## Author

**Rana Ali Husnain**

https://github.com/Rana-Ali-Husnain
