import requests


def query_qwen(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3.5:9b", 
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": 256, "temperature": 0.8},
        },
    )
    return response.json()["response"]


if __name__ == "__main__":
    fixed_prompt = """Answer the question in ONE WORD only.
Do not explain.
Do not think.
Just output the final answer. 


Question: What is the capital City of australia
Answer:"""
    res = query_qwen(fixed_prompt)
    print(res)
