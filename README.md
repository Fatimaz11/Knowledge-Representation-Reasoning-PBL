# Fuzzy Weather Comfort System

A Python-based fuzzy logic system to predict weather comfort index based on temperature and humidity using **Mamdani fuzzy inference**. This project includes unit tests, visualization tools, and a working frontend for real-time interaction.

---

## Features

- **Predict Comfort Index**: Computes a comfort score (0-100) and textual level (Uncomfortable, Neutral, Comfortable) from temperature and humidity.
- **Batch Processing**: Process an entire weather dataset (`CSV`) and generate predictions.
- **Visualizations**:
  - Membership function plots
  - Comfort heatmap across temperature and humidity ranges
- **Unit Tests**: Ensure correctness of membership functions, rule evaluation, and output consistency.
- **Frontend Interface**: Simple web app using Flask to input temperature and humidity and get instant predictions.

---

## Requirements

- Python 3.10+ recommended
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-fuzzy`
  - `scikit-learn` (optional for evaluation)
  - `flask` (for frontend)

Install all dependencies using:

```bash
pip install -r requirements.txt
