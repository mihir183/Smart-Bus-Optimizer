import os
import sys
import time
from app import create_app, get_socketio
from app.simulator import BusSimulator
import logging

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def simulate_loading():
    """Simulate loading by introducing delays."""
    logger.info("Loading real-time data... Please wait.")
    time.sleep(5)  # Simulate a 5-second loading time
    logger.info("Data loaded successfully!")

def main():
    """Main function to run the application"""
    try:
        # Create Flask app
        app = create_app()
        socketio = get_socketio()
        
        # Simulate loading
        simulate_loading()

        # Initialize simulator within app context
        with app.app_context():
            simulator = BusSimulator(socketio)
            # Start simulation for all routes
            simulator.start_simulation()

        logger.info("Starting Smart Bus System...")
        logger.info("Dashboard available at: http://localhost:5000")
        logger.info("API documentation available at: http://localhost:5000/api/health")
        
        # Run the application
        socketio.run(
            app,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_ENV') == 'development',
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down Smart Bus System...")
        if 'simulator' in locals():
            simulator.stop_simulation()
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    # Run the application
    main()
