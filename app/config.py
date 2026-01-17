import os

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if HF_TOKEN is None:
    raise EnvironmentError(
        "HUGGINGFACEHUB_API_TOKEN is not set. "
        "Please define it as an environment variable."
    )
