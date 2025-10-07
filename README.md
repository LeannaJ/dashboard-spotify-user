# 🎵 Spotify User Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing Spotify user behavior patterns and churn prediction.

## 📊 Overview

This dashboard provides deep insights into Spotify user engagement, demographics, and churn patterns through interactive visualizations and analytics.

## 🎯 Features

### 📈 **5 Main Analysis Sections**
- **Overview**: User distribution and subscription patterns
- **User Demographics**: Gender, age, and geographic analysis
- **Engagement Analysis**: Listening behavior and usage patterns
- **Churn Analysis**: Churn rate analysis and risk factors
- **Advanced Analytics**: Correlation analysis and predictive insights

### 🎛️ **Interactive Features**
- Real-time filtering by gender, age group, subscription type, and country
- Dynamic KPI updates based on selected filters
- Interactive charts with Plotly
- Responsive design optimized for all devices

### 📊 **Key Metrics Tracked**
- Total users and churn rates
- Average listening time and songs per day
- Skip rates and engagement patterns
- Premium vs Free user analysis
- Device usage patterns

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data-analytics-dashboard.git
cd data-analytics-dashboard/Spotify\ User\ Analytics\ Dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run streamlit_spotify.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
Spotify User Analytics Dashboard/
├── streamlit_spotify.py          # Main dashboard application
├── spotify_churn_tidy.csv        # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🎨 Design

The dashboard features Spotify's signature green color scheme (#1DB954) with:
- Modern, clean interface
- Responsive design
- Interactive visualizations
- Brand-consistent styling

## 📊 Dataset

The dashboard uses the `spotify_churn_tidy.csv` dataset which includes:
- User demographics (gender, age, country)
- Subscription information
- Listening behavior metrics
- Device usage patterns
- Churn indicators

## 🔧 Technologies Used

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Seaborn**: Statistical visualizations
- **Matplotlib**: Plotting library

## 🌐 Deployment

### Streamlit Cloud
This dashboard can be deployed on Streamlit Cloud:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Select the main file (`streamlit_spotify.py`)
4. Deploy!

### Local Deployment
For local deployment, simply run:
```bash
streamlit run streamlit_spotify.py
```

## 📈 Key Insights

The dashboard provides insights into:
- **User Engagement**: Listening patterns and behavior
- **Churn Prediction**: Risk factors and prevention strategies
- **Market Segmentation**: Demographics and preferences
- **Subscription Optimization**: Premium conversion opportunities

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this dashboard.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created with ❤️ for data analytics and visualization enthusiasts.

---

**🎵 Built with Python, Streamlit & Plotly**
