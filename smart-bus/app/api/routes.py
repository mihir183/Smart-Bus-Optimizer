from .. import api_bp

@api_bp.route('/health', methods=['GET'])
def health_check():
    return {'status': 'ok', 'message': 'Smart Bus API is running'}

@api_bp.route('/hello', methods=['GET'])
def hello():
    return {'message': 'Hello from Smart Bus API'}
