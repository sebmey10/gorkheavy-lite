from llama_swap.swap_script import models

ROLE_KEYS = {"promptimizer", "judge"}  # models we don't want evaluated

EXPERIMENT_MODEL_NAMES = [j for i, j in models.items() if i not in ROLE_KEYS] # [key,value]
print(EXPERIMENT_MODEL_NAMES)
