# Directories
DATA_DIR = "data"
HPO_OUTPUT_DIR = "hpo_results"
MODELS_DIR = "models"

# Optimization output files paths
# first {} is the identifier for kan or ann, second is for the target (laminar flame speed or density ratio)
HPO_CSV_FILENAME = "hpo_results_{kan_or_ann}_{target}.csv" 
HPO_BEST_PARAMS_FILENAME = "best_params_{kan_or_ann}_{target}.json"
HPO_LOG_FILENAME = "hpo_log_{kan_or_ann}_{target}.txt"

# NN model file name
MODEL_FILENAME = "{kan_or_ann}_{target}_model.pt"
