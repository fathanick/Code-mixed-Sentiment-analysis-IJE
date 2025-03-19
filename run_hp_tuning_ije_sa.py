import subprocess
from tqdm import tqdm

# Define common arguments
common_args = "--output_dir best_parameters --num_trials 5"

# Define the list of commands with model names and corresponding JSON files
commands = [
    f"python hp_tuning_ije_sa.py --model_name fathan/indojave-codemixed-bert-base {common_args} --save_best_params best_parameters/indojave-cm-bert.json",
    f"python hp_tuning_ije_sa.py --model_name fathan/indojave-codemixed-indobert-base {common_args} --save_best_params best_parameters/indojave-cm-indobert.json",
    f"python hp_tuning_ije_sa.py --model_name fathan/indojave-codemixed-indobertweet-base {common_args} --save_best_params best_parameters/indojave-cm-indobertweet.json",
    f"python hp_tuning_ije_sa.py --model_name fathan/indojave-codemixed-roberta-base {common_args} --save_best_params best_parameters/indojave-cm-roberta.json",
    f"python hp_tuning_ije_sa.py --model_name indolem/indobert-base-uncased {common_args} --save_best_params best_parameters/indobert.json",
    f"python hp_tuning_ije_sa.py --model_name indolem/indobertweet-base-uncased {common_args} --save_best_params best_parameters/indobertweet.json",
    f"python hp_tuning_ije_sa.py --model_name bert-base-multilingual-uncased {common_args} --save_best_params best_parameters/mbert.json",
    f"python hp_tuning_ije_sa.py --model_name xlm-roberta-base {common_args} --save_best_params best_parameters/xlm-roberta.json",
    f"python hp_tuning_ije_sa.py --model_name distilbert-base-multilingual-cased {common_args} --save_best_params best_parameters/distilmbert.json"
]

# Execute the commands with progress bars
for command in tqdm(commands, desc="Running commands"):
    subprocess.run(command, shell=True)

print("All commands executed successfully.")
