from locust import HttpUser, task, between

class FastAPIUser(HttpUser):
    # Wait time between tasks (1 to 3 seconds)
    wait_time = between(1, 3)

    @task(1)
    def predict(self):
        # Define the payload for the /predict endpoint
        text = "I really hate working in a place that is larger than any place"

        # Send a POST request to the /predict endpoint
        self.client.post("/predict", json={"text":text})

    @task(2)
    def predict_quantized(self):
        # Define the payload for the /generate endpoint
        text = "I really hate working in a place that is larger than any place"

        # Send a POST request to the /generate endpoint
        self.client.post("/predict_quantized", json={"text":text})


