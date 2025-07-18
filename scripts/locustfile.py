from locust import HttpUser, task

class InferenceUser(HttpUser):
    @task
    def predict_tile(self):
        with open("test_assets/sample_tile.jpg", "rb") as img:
            self.client.post("/predict", files={"file": img})
