from api.interp_inference import DEFAULT_MODEL, InterpInference


def main():
    inference = InterpInference(model=DEFAULT_MODEL)

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
