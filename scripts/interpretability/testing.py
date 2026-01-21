from torch import save

from api.interp_inference import InterpInference

SMALL_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


def save_logits(model):
    """Intervention that saves the logits."""
    return model.lm_head.output[0][-1].detach().cpu()


def main():
    inference = InterpInference(model=SMALL_MODEL)

    response = inference.generate_from_messages(
        messages=[
            {
                "role": "user",
                "content": "What is the difference between a cat and a dog?",
            }
        ],
        max_new_tokens=100,
        intervention=save_logits,
    )

    print(f"Response: {response.content}")

    save(response.intervention_output["logits"], "logits.pt")


if __name__ == "__main__":
    main()
