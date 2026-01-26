import json
from typing import Dict, List

from torch import save

from api.interp_inference import InterpInference

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

SYSTEM_PROMPT = "You are a helpful assistant participating in a conversation."

with open("data/sharegpt_sample.json", "r") as f:
    conversations = json.load(f)

conversation = conversations[0]
rounds = 50


def define_msg_tree(
    messages: List[Dict[str, str]],
    system_prompt: str | None = None,
) -> List[Dict[str, str]]:
    """
    Canonicalize conversation into a stable chat format:
    - Optional system prompt first
    - Strict user/assistant alternation
    - Last message is always `user`
    """
    tree = []

    if system_prompt:
        tree.append({"role": "system", "content": system_prompt})

    # Reassign roles based on recency
    rewritten = []
    for i, message in enumerate(reversed(messages)):
        role = "user" if i % 2 == 0 else "assistant"
        rewritten.append({"role": role, "content": message["content"]})

    tree.extend(reversed(rewritten))
    return tree


def save_logits(model):
    """Intervention that saves the logits."""
    return model.lm_head.output[0][-1].detach().cpu()


def main():
    inference = InterpInference(model=DEFAULT_MODEL)
    logits_list = []

    rounds_left = rounds - len(conversation)
    for _ in range(rounds_left):
        messages = define_msg_tree(
            conversation,
            system_prompt=SYSTEM_PROMPT,
        )

        print(f"Messages: {messages}")

        response = inference.generate_from_messages(
            messages=messages,
            max_new_tokens=256,
            temperature=0.7,
            top_p=1.0,
            intervention=save_logits,
        )

        conversation.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        print(f"Response: {response.content}")
        logits_list.append(response.intervention_output["logits"])

    save(logits_list, "logits.pt")
    with open("conversation.json", "w") as f:
        json.dump(conversation, f, indent=2)


if __name__ == "__main__":
    main()
