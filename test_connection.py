from ray import serve
from fastapi import FastAPI

serve.start(detached=True, http_options={"host": "0.0.0.0", "port": "8000"})
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class HelloWorld:
    @app.get("/")
    def say_hello(self):
        return {"message": "Hello, Ray Serve!"}

# Define the deployment entry point
hello_world = HelloWorld.bind()
