from locust import HttpUser, task, between

class MLUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_image(self):
        with open("data/test/sample.jpg", "rb") as f:
            self.client.post("/predict/", files={"file": f})
