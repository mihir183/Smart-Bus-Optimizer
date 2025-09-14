from flask import Blueprint, jsonify
from sqlalchemy import text
 # Import 'db' inside the function to avoid circular import

# Create blueprint for health check
health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint.
    Returns "healthy" if DB connection works, otherwise "unhealthy".
    """
    try:
        # Import here to avoid circular import
        from app import db
        db.session.execute(text("SELECT 1"))
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
