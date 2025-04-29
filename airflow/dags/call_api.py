import requests

def call_app_api():
    """
    Call the API to trigger the retraining service.
    """
    try:
        response = requests.get("http://app:8000/retrain/")
        print(response.status_code, response.text)
    except Exception as e:
        print(f"Error calling API: {e}")
