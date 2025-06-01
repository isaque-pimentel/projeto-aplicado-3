import os

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'histflix-secret')
    # Add other config variables here as needed
