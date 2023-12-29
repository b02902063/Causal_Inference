import requests
import os
from datetime import datetime
import json
import numpy as np
import base64

url = 'https://asia-east1-ameai-causal.cloudfunctions.net/inference'



J = {
    "model_id": "gs://causal_data/Discovery_data/Rayark/sdorica_player.csv",
    "before": {"currency_freeGem": 0},
    "after": {"currency_freeGem": 20000}
}

headers = {
    'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjQ1NmI1MmM4MWUzNmZlYWQyNTkyMzFhNjk0N2UwNDBlMDNlYTEyNjIiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF6cCI6ImFtZWFpLWNhdXNhbEBhcHBzcG90LmdzZXJ2aWNlYWNjb3VudC5jb20iLCJlbWFpbCI6ImFtZWFpLWNhdXNhbEBhcHBzcG90LmdzZXJ2aWNlYWNjb3VudC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzAzODY0MTYxLCJpYXQiOjE3MDM4NjA1NjEsImlzcyI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbSIsInN1YiI6IjExMjE5NTI3Njg3MzMzODAwMzA2NiJ9.E1OeMbfpDG1jJ-m2L8yX2f52mo9QOkoJJ3gkT9cuU6lqIbbMkFprH-YNAbJ88HM4g_8tI-K2GYvjUYoY-l2pULZT-do_D2ffoLjkxM9dQ7jKtHwrQKGDpZu3vGFtWyPy6pJEbAek8FdNw9c0iSbaZHKZYZXmwyI97_7lZ1HWmG_tx_9t5jW19i3uBMtKrP2ZcjX-WleTypP06DXWGRd6yIGi1f6_xM6NG9L-2FrAFRX0NK7oHoc8tk3d8x-IxkRAi7PaZoT4MlrCD2LmLq_-qSVTpRRKOAjzMqly22u-pWhFlNAfOHB_I-2Qn4moTbThJm4nu5e-vg6ImbrM23KBnQ',
}

response = requests.post(
    url,
    headers=headers,
    json=J
)

print(response.status_code) 

J = response.json()
print(J)  