import os
import firebase_admin
from firebase_admin import credentials, db
from pathlib import Path

def test_firebase_connection():
    try:
        # Get the absolute path to the config file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'firebase-key.json')
        
        # Check if the config file exists
        if not os.path.exists(config_path):
            print(f"❌ Error: Firebase key not found at {config_path}")
            print("Please download the service account key from Firebase Console and save it as 'firebase-key.json' in the config folder.")
            return False
        
        # Initialize Firebase
        cred = credentials.Certificate(config_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://potholesdb-default-rtdb.firebaseio.com/'
        })
        
        # Test database connection
        ref = db.reference('/')
        print("✅ Successfully connected to Firebase")
        print(f"Database URL: {ref.path}")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Firebase: {str(e)}")
        return False

if __name__ == "__main__":
    test_firebase_connection()
