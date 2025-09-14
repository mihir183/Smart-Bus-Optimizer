# Smart Bus System 🚌

A comprehensive intelligent bus management system with real-time tracking, predictive analytics, and automated scheduling optimization.

## 🌟 Features

### Core Functionality
- **Real-time Bus Tracking**: Live GPS updates and bus location monitoring
- **Predictive Analytics**: ML-powered arrival time predictions
- **Intelligent Scheduling**: Automated bunching detection and schedule optimization
- **Occupancy Monitoring**: Real-time passenger count tracking
- **Performance Analytics**: Comprehensive route and service performance metrics

### Technical Features
- **RESTful API**: Complete API for data access and system control
- **Real-time Updates**: WebSocket-based live data streaming
- **Interactive Dashboard**: Modern web interface with real-time maps
- **Data Pipeline**: Automated data cleaning and quality monitoring
- **Machine Learning**: Scikit-learn based prediction models
- **Docker Support**: Containerized deployment with Docker Compose

## 🏗️ Architecture

```
smart-bus/
├── app/                    # Main application code
│   ├── __init__.py        # Flask app factory and SocketIO setup
│   ├── models.py          # SQLAlchemy database models
│   ├── api.py             # REST API endpoints
│   ├── scheduler.py       # Intelligent scheduling engine
│   ├── simulator.py       # Real-time bus simulation
│   ├── predictor.py       # ML prediction models
│   ├── cleaner.py         # Data cleaning pipeline
│   ├── utils.py           # Utility functions and metrics
│   ├── static/            # Static web assets
│   │   ├── css/           # Stylesheets
│   │   ├── js/            # JavaScript application
│   │   └── libs/          # Third-party libraries
│   └── templates/         # HTML templates
├── data/                  # Data storage
│   ├── raw/               # Raw data files
│   ├── clean/             # Cleaned data files
│   └── generate_synthetic.py  # Synthetic data generator
├── notebooks/             # Jupyter notebooks for analysis
├── Dockerfile             # Docker container configuration
├── docker-compose.yml     # Multi-service Docker setup
├── requirements.txt       # Python dependencies
├── run.py                 # Application entry point
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (optional)
- PostgreSQL (if not using Docker)

### Installation

#### Option 1: Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd smart-bus

# Start all services
docker-compose up -d

# Generate synthetic data
docker-compose exec smart-bus python data/generate_synthetic.py

# Access the application
open http://localhost:5000
```

#### Option 2: Local Development (Automated Setup)
```bash
# Clone the repository
git clone <repository-url>
cd smart-bus

# Run automated setup (creates venv, installs deps, generates data)
python setup.py

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run the application
python run.py
```

#### Option 3: Manual Local Development
```bash
# Clone the repository
git clone <repository-url>
cd smart-bus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (SQLite version for local development)
pip install -r requirements.txt

# For production with PostgreSQL, use:
# pip install -r requirements-prod.txt

# Set environment variables
export DATABASE_URL="sqlite:///smart_bus.db"
export SECRET_KEY="your-secret-key"

# Generate synthetic data
python data/ge  

# Run the application
python run.py
```

## 📊 Usage

### Web Dashboard
- **Main Dashboard**: http://localhost:5000
- **Map View**: http://localhost:5000/map
- **Route Details**: http://localhost:5000/routes

### API Endpoints
- **Health Check**: `GET /api/health`
- **Routes**: `GET /api/routes`
- **Buses**: `GET /api/buses`
- **Trips**: `GET /api/trips`
- **Predictions**: `GET /api/predictions`
- **Generate Predictions**: `POST /api/predictions/generate`
- **Detect Bunching**: `POST /api/schedule/detect-bunching`

### Jupyter Notebooks
- **Data Analysis**: http://localhost:8888 (if using Docker)
- **Model Development**: `notebooks/eda_model.ipynb`

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/smartbus

# Security
SECRET_KEY=your-secret-key-change-in-production

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Application
FLASK_ENV=development  # or production
PORT=5000
```

### Database Models
- **Routes**: Bus route information and stops
- **Buses**: Individual bus details and capacity
- **Trips**: Scheduled and actual trip data
- **Events**: Real-time GPS, occupancy, and delay events
- **Predictions**: ML model predictions
- **Schedule Adjustments**: Automated schedule changes

## 🤖 Machine Learning

### Prediction Models
- **Arrival Time Prediction**: Random Forest and Gradient Boosting models
- **Feature Engineering**: Time-based, route-based, and historical features
- **Model Training**: Automated training pipeline with cross-validation
- **Performance Monitoring**: Real-time accuracy tracking

### Data Pipeline
- **Data Cleaning**: Automated outlier detection and data quality checks
- **Feature Engineering**: Time series features and route characteristics
- **Model Deployment**: Automated model updates and versioning

## 📈 Analytics

### Key Metrics
- **On-time Performance**: Percentage of trips within 5 minutes of schedule
- **Average Delays**: Mean delay across all trips
- **Occupancy Rates**: Average and peak passenger loads
- **Bunching Events**: Frequency and severity of bus bunching
- **Route Efficiency**: Performance scores for each route

### Real-time Monitoring
- **Live Bus Tracking**: Real-time GPS updates
- **Occupancy Monitoring**: Passenger count tracking
- **Delay Alerts**: Automatic delay notifications
- **Performance Dashboards**: Live metrics and KPIs

## 🛠️ Development

### Code Structure
- **Models**: SQLAlchemy ORM models for database entities
- **API**: RESTful endpoints with proper error handling
- **Services**: Business logic for scheduling and predictions
- **Frontend**: Modern JavaScript with real-time updates
- **Testing**: Comprehensive test suite with pytest

### Adding New Features
1. **Database Changes**: Update models in `app/models.py`
2. **API Endpoints**: Add routes in `app/api.py`
3. **Business Logic**: Implement in appropriate service modules
4. **Frontend**: Update JavaScript in `app/static/js/`
5. **Templates**: Modify HTML in `app/templates/`

### Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_api.py
```

## 🚀 Deployment

### Production Deployment
```bash
# Build production image
docker build -t smart-bus:latest .

# Run with production settings
docker run -d \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  -e DATABASE_URL=postgresql://... \
  smart-bus:latest
```

### Scaling
- **Horizontal Scaling**: Multiple app instances behind load balancer
- **Database**: PostgreSQL with read replicas
- **Caching**: Redis for session storage and caching
- **Monitoring**: Application performance monitoring

## 📚 API Documentation

### Authentication
Currently, the API is open for development. In production, implement proper authentication.

### Rate Limiting
API endpoints are rate-limited to prevent abuse. Default limits:
- 100 requests per minute per IP
- 1000 requests per hour per IP

### Error Handling
All API endpoints return consistent error responses:
```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

## 🔍 Monitoring and Logging

### Logging
- **Application Logs**: `logs/app.log`
- **Error Tracking**: Structured logging with context
- **Performance Metrics**: Request timing and database queries

### Health Checks
- **System Health**: `/api/health` endpoint
- **Database Connectivity**: Automatic health monitoring
- **Service Dependencies**: Redis and external service checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **Scikit-learn**: Machine learning library
- **Socket.IO**: Real-time communication
- **Leaflet**: Interactive maps
- **Bootstrap**: UI framework

## 📞 Support

For support and questions:
- **Issues**: GitHub Issues
- **Documentation**: This README and inline code comments
- **Email**: [your-email@example.com]

## 🔮 Roadmap

### Upcoming Features
- [ ] Mobile app for passengers
- [ ] Advanced analytics dashboard
- [ ] Integration with external transit APIs
- [ ] Machine learning model improvements
- [ ] Real-time passenger notifications
- [ ] Advanced scheduling algorithms
- [ ] Multi-city support
- [ ] API versioning

### Performance Improvements
- [ ] Database query optimization
- [ ] Caching layer implementation
- [ ] Real-time data streaming optimization
- [ ] Machine learning model optimization

---

**Smart Bus System** - Making public transportation smarter, more efficient, and more reliable. 🚌✨
