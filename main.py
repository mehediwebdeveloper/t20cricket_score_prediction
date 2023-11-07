from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
import numpy as np
from urllib.parse import urlparse, parse_qs
import cgi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import pandas as pd
dataset = pd.read_csv('ipl.csv')
X = dataset.iloc[:,[7,8,9,12,13]].values #Input features
y = dataset.iloc[:, 14].values #Label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)

def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

y_pred = lin.predict(X_test)
score = lin.score(X_test,y_test)*100
print("R-squared value:" , score)
print("Custom accuracy:" , custom_accuracy(y_test,y_pred,10))


class PredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve the HTML form
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = """
       <html>
<head>
    <title>Cricket Score Prediction</title>
    <style>
        /* Add your internal CSS styles here */
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            margin: 0;
            padding: 0;
        }

        h1 {
            text-align: center;
        }

        form {
            background-color: #ffffff;
            width: 300px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        }

        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        input[type="submit"] {
            background-color: #007bff;
            color: #fff;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        input[type="submit"]:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <h1>T20 Cricket Score Prediction</h1>
    <form method="POST">
        Runs: <input type="text" name="runs"><br>
        wickets: <input type="text" name="wickets"><br>
        Overs: <input type="text" name="overs"><br>
        Striker: <input type="text" name="striker"><br>
        Non-Striker: <input type="text" name="non-striker"><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>

        """
        self.wfile.write(html.encode())

    def do_POST(self):
        # Parse the form data
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )

        # Extract values from form fields
        runs = float(form["runs"].value)
        wickets = float(form["wickets"].value)
        overs = float(form["overs"].value)
        striker = float(form["striker"].value)
        non_striker = float(form["non-striker"].value)

        # Predict using the loaded model
        new_prediction = lin.predict(sc.transform(np.array([[runs,wickets, overs, striker, non_striker]])))

        # Prepare the response
        response = f"Prediction score: {new_prediction[0]}"

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(response.encode())

# Create and run the HTTP server
if __name__ == '__main__':
    server_address = ('', 8000)  # You can change the port if needed
    httpd = HTTPServer(server_address, PredictionHandler)
    print("Web server is running...")
    httpd.serve_forever()
