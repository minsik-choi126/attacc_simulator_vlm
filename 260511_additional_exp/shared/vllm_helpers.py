"""Shared vLLM helpers for VLM measurement scripts."""


def detect_template_text(model_name, prompt):
    """Return a chat-templated prompt string with one image placeholder."""
    name = model_name.lower()
    if "qwen3-vl" in name or "qwen2.5-vl" in name or "qwen2-vl" in name:
        return ("<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                "{}<|im_end|>\n<|im_start|>assistant\n").format(prompt)
    if "llava-1.5" in name or "llava-v1.5" in name:
        return "USER: <image>\n{} ASSISTANT:".format(prompt)
    if "llava-v1.6" in name or "llava-1.6" in name or "llava-next" in name:
        return "[INST] <image>\n{} [/INST]".format(prompt)
    if "internvl" in name:
        # InternVL*-hf use <IMG_CONTEXT> (not <image>) as the placeholder vLLM
        # expands; <image> makes vLLM 0.17 raise "Failed to apply prompt
        # replacement". Matches the HF processor chat template exactly.
        return ("<|im_start|>user\n<IMG_CONTEXT>\n{}<|im_end|>\n"
                "<|im_start|>assistant\n").format(prompt)
    return prompt


def make_image_input(model_name, prompt, image):
    return {
        "prompt": detect_template_text(model_name, prompt),
        "multi_modal_data": {"image": image},
    }
