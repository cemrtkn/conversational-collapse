from api.interp_inference import InterpInference

def main():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    inference = InterpInference(model_id=model_id)

    response = inference.generate_from_messages(
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        max_new_tokens=100,
    )
    print(response.content)


if __name__ == "__main__":
    main()