# SmartSure: Revolutionizing Insurance with AI
### Empowering a healthier you, smarter insurance too. 

![](https://raw.githubusercontent.com/En1gma02/SmartSure/refs/heads/main/images/SmartSure_logo.png)

Welcome to **SmartSure**, an innovative project designed to transform the insurance industry using cutting-edge AI technologies. This project was developed by Akshansh Dwivedi for the **Finovate Hacks** competition.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
  - [AI-Powered Customer Support](#ai-powered-customer-support)
  - [Fitness Score Calculator](#fitness-score-calculator)
  - [AI-Driven Base Plan Recommendations](#ai-driven-base-plan-recommendations)
  - [Customizable Insurance Plan Generator](#customizable-insurance-plan-generator)
- [Installation](#installation)
- [Usage](#usage)
- [Active Contributor](#active-contributor)
- [Contributing](#contributing)
- [License](#license)

## Introduction

SmartSure is a comprehensive platform that leverages artificial intelligence to provide personalized insurance solutions. It offers a range of features designed to enhance the user experience, from AI-powered customer support to dynamic plan generation.

## Features

### AI-Powered Customer Support

- **Technology Used:** Mistral and Hugging Face API for natural language processing.
- **Datasets:** Trained on extensive datasets, including 100+ insurance QA pairs and a dataset with 50,000 rows provided by YouData.ai.
- **Performance:** Achieves 95% accuracy in query resolution.
- **Impact:** Significantly reduces customer service response time, ensuring swift and efficient support.

### Fitness Score Calculator and Certificate Generator

- **Technology Used:** Random Forest AI-ML Model.
- **Datasets:** Trained on over 50,000 health and claims data points from a YouData.ai dataset.
- **Performance:** Achieves 92% accuracy in predicting health risks.
- **Impact:** Factors in 15+ health metrics, such as blood pressure, BMI, and sleep quality, to offer up to 30% premium discounts based on fitness levels.
- **Certificate:** Provides a detailed, personalized Fitness Certificate with 10+ parameters.

### AI-Driven Base Plan Recommendations

- **Technology Used:** K-Nearest Neighbors (KNN) AI-ML Model.
- **Datasets:** Trained on 60,000+ insurance records from a YouData.ai dataset.
- **Performance:** Provides personalized recommendations with 88% accuracy in plan matching.
- **Impact:** Considers 10+ user parameters, including age, income, and occupation, offering interactive data visualizations for trend analysis.

### Customizable Insurance Plan Generator

- **Technology Used:** Neural Networks AI-ML Model with 3 layers.
- **Datasets:** Utilizes data from a YouData.ai dataset.
- **Performance:** Provides dynamic pricing with 95% accuracy compared to traditional actuarial methods.
- **Impact:** Generates personalized plans in under 5 seconds based on 6 key parameters: age, gender, profession, fitness score, insurance type, and coverage amount.

## Installation

To install and run SmartSure locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/En1gma02/SmartSure
    cd SmartSure
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    ```bash
    streamlit run main.py
    ```

## Usage

Once the application is running, you can explore the following features:

-   **AI-Powered Customer Support:** Ask any insurance-related question and receive instant, accurate responses.
-   **Fitness Score Calculator:** Input your health metrics to calculate your fitness score and potential premium discounts.
-   **AI-Driven Base Plan Recommendations:** Get personalized insurance plan recommendations based on your user profile.
-   **Customizable Insurance Plan Generator:** Tailor an insurance plan to your specific needs and receive dynamic pricing instantly.
-   **Dynamic Business Dashboard:** Access real-time sales insights tailored to your chosen parameters, along with their respective visualizations.

### Active Contributor 
<hr>
<p align="start">
<a  href="https://github.com/En1gma02/SmartSure/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=En1gma02/SmartSure"/>
</a>
</p>
</br>

## Contributing

Contributions to SmartSure are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
