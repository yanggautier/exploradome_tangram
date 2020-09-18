import requests
res = requests.get("http://localhost:5000/multiply?x={0}".format(2.0))
data = res.json()
print("multiply result : ", data['result'])
r = requests.post("http://localhost:5000/add", json={'x': 2.5, 'y':3.5})
print("add result", r.json()["result"])
print(r.status_code, r.reason)