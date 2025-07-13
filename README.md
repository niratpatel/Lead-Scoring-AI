# AI Lead Scoring - Full-Stack Application

### A production-ready, deployed system for intelligently prioritizing sales leads.

<br>

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=githubpages)](https://niratpatel.github.io/Lead-Scoring-AI/)
[![Backend API](https://img.shields.io/badge/Backend-API-blueviolet?style=for-the-badge&logo=render)](https://lead-scorer-api.onrender.com/docs)
[![Technology](https://img.shields.io/badge/Stack-Full%20Stack-blue?style=for-the-badge)]()

---

This is not just a model; it's a complete, end-to-end web application that solves a critical business problem: **sales teams waste significant time and resources pursuing low-quality leads.** This project leverages a sophisticated machine learning pipeline to analyze and score incoming leads, allowing sales teams to focus their efforts on high-intent prospects who are most likely to convert.

The entire application is deployed with a modern, decoupled architecture and is accessible at the link above.

## Key Features

-   **Real-Time AI Scoring:** A user-friendly form that provides an instant, AI-generated intent score upon submission.
-   **Contextual Reranking:** A rule-based system analyzes user comments (e.g., "ready to buy immediately") to adjust the ML score, capturing explicit intent that a statistical model would miss.
-   **Interactive Dashboard:** A dynamic chart visualizes the distribution of lead quality, providing an at-a-glance overview for sales managers.
-   **Persistent Sessions:** Scored leads are saved to the browser's `localStorage`, so the data persists even after a page refresh.
-   **Fully Deployed:** A complete CI/CD pipeline with a FastAPI backend on Render and a high-performance frontend on GitHub Pages, accessible via a public URL with low latency.

## Tech Stack & Architecture

This project was built with a professional, decoupled architecture to ensure scalability and maintainability.

| Category           | Technology                    | Purpose                                                                |
| :----------------- | :---------------------------- | :--------------------------------------------------------------------- |
| **Frontend**       | HTML5, CSS3, Vanilla JavaScript | For a fast, lightweight, and universally compatible user interface.    |
|                    | Chart.js                      | To create beautiful, interactive data visualizations.                  |
|                    | GitHub Pages                  | For reliable, high-speed hosting of the static frontend application.   |
| **Backend**        | Python 3                      | The core programming language for the API and ML logic.                |
|                    | FastAPI                       | A modern, high-performance web framework for building the RESTful API. |
|                    | Pydantic                      | For robust, automatic data validation at the API level.                |
|                    | Render                        | For scalable, cloud-based hosting of the Python backend.               |
| **Machine Learning** | Scikit-learn                  | For building the robust data preprocessing and modeling pipeline.      |
|                    | LightGBM                      | The industry-standard gradient boosting model for high performance.    |
|                    | Joblib                        | For serializing and deserializing the trained ML pipeline object.      |

### System Architecture Flow

A user's request flows through a modern, decoupled system:

`User Browser` → `GitHub Pages (HTML/CSS/JS)` → `Render (FastAPI Backend)` → `ML Model & Reranker` → `JSON Response`

## The Machine Learning Core

The heart of the application is a pipeline designed to extract a reliable predictive signal from a noisy, real-world dataset.

1.  **Feature Engineering:** The pipeline begins by creating high-impact features from raw data, such as `days_since_creation` (to capture lead decay) and `submission_to_view_ratio` (to measure user intent). It also uses context-aware imputation, for example, by filling a missing income value with the median income of the lead's stated profession.

2.  **The Model: Why LightGBM?** After initial experiments, **LightGBM** (`LGBMClassifier`) was chosen for its optimal balance of speed and power. It is the industry standard for production systems working with tabular data because its unique leaf-wise growth algorithm is significantly faster than alternatives, ensuring the API meets its strict low-latency requirements. Its powerful built-in regularization also makes it highly effective at handling the noisy features present in the dataset.

3.  **The Reranker:** A rule-based module provides a crucial final layer of business logic. It scans user comments for high-intent keywords ("urgent", "schedule a call") and negative keywords ("just browsing", "too expensive") to make a final adjustment to the score. This hybrid approach—combining statistical patterns with explicit user intent—creates a final score that is both more accurate and more interpretable.

## Challenges & Solutions

Building and deploying a full-stack ML application presents real-world challenges. Overcoming them was a key part of this project.

### Challenge 1: The ML Deployment "Catch-22"

-   **Problem:** The most significant hurdle was a recurring `AttributeError` during deployment. The FastAPI backend could not load the saved `joblib` model because the file was "pickled" with a reference to a custom Python class (`HighPerformanceLeadScoringModel`) that was defined in the training script, not the API's runtime environment.
-   **Solution:** I solved this by refactoring the project to professional standards. I created a dedicated **`model_loader.py`** file to house the custom class definitions. I then wrote a new, separate training script, **`create_model.py`**, which **imports the class from `model_loader` before saving the model**. This permanently "baked" the correct import path into the `.pkl` file. The FastAPI app now also imports from `model_loader`, allowing it to correctly interpret and load the model artifact without error. This demonstrates a robust, professional pattern for deploying complex ML models.

### Challenge 2: Full-Stack Deployment & Configuration

-   **Problem:** The initial deployment attempts failed due to subtle but critical configuration mismatches between the local environment and the cloud platforms. This included Render requiring an explicit `email-validator` dependency and initial frontend deployment platforms struggling with git branch name mismatches.
-   **Solution:** I systematically debugged the entire CI/CD pipeline. I corrected the `requirements.txt` file to include all necessary sub-dependencies. When the initial platform (Netlify) proved problematic, I pivoted to a more direct and foolproof deployment strategy using **GitHub Pages**. This involved re-configuring the project structure to use a `/docs` folder and updating the backend's CORS policy on Render to securely accept requests from the new domain. This process showed an ability to adapt and solve end-to-end deployment issues across multiple cloud services.

## (How to Run Locally)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/niratpatel/Lead-Scoring-AI.git
    cd Lead-Scoring-AI
    ```

2.  **Setup Python Environment:**
    ```bash
    python -m venv venv
    source ven/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Create the Model File:**
    -   First, you need the dataset (`lead_scoring_data_enhanced.csv`) in your root folder.
    -   Run the training script to generate the `high_performance_model.pkl` file:
        ```bash
        python create_model.py
        ```

4.  **Run the Backend:**
    ```bash
    uvicorn app:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

5.  **Run the Frontend:**
    -   Navigate to the `/docs` folder.
    -   Open the `index.html` file in your web browser.
