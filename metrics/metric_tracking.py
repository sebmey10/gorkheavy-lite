test_cases = {
    "promptimizer": "Olmo-3-7B-Instruct-Q8_0",
    "llama": "llama3.1-8B-Q8_0",
    "gemma": "gpt-oss-20b-F16",
    "gemma_small": "Olmo-3-7B-Think-Q8_0",
    "judge": "granite-3.2-8b-instruct-f16",
    "devstral": "Devstral-Small-2-24B-Instruct-2512-UD-Q5_K_XL",
    "exaone": "EXAONE-4.0-32B-Q4_K_M",
    "lfm_extract": "LFM2-1.2B-Extract-Q8_0",
    "lfm_rag": "LFM2-1.2B-RAG-Q8_0",
    "lfm_tool": "LFM2-1.2B-Tool-Q8_0",
    "olmo_instruct": "Olmo-3-7B-Instruct-Q8_0",
    "olmo_think": "Olmo-3-7B-Think-Q8_0",
    "gemma_12b": "gemma3-12b-Q6_K",
    "gemma_27b": "gemma-3-27b-it-Q6_K",
    "gpt_oss_20b": "gpt-oss-20b-F16",
    "gpt_oss_120b": "gpt-oss-120b-F16",
    "mistral": "mistral3.2-24B-Q6_K"
}


ROLE_KEYS = {"promptimizer", "judge"}  # models we don't want evaluated

EXPERIMENT_MODEL_NAMES = [j for i, j in test_cases.items() if i not in ROLE_KEYS] # [key,value]
print(EXPERIMENT_MODEL_NAMES)
