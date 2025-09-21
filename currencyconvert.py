import requests
from datetime import date

def get_conversion_rate(base: str, target: str):
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{today}/v1/currencies/{base.lower()}.json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data[base.lower()].get(target.lower())
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example: 1 INR = ? USD
# rate = get_conversion_rate("inr", "usd")
# print("1 INR =", rate, "USD")

