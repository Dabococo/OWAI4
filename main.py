import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Verifier si CUDA est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger la configuration
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Specifiez le chemin local vers le repertoire du modele (pas le fichier .bin directement)
chemin_local = "Dabococo/owai_model"

# Charger le modele fine-tune a partir du chemin local et le deplacer sur le GPU
model = AutoModelForCausalLM.from_pretrained(chemin_local, config=config).to(device)

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Tokeniser le texte d'entree et le deplacer sur le GPU
input_text = "Hello what is your name ?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generer du texte sur le GPU
output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

# Deplacer la sortie sur le CPU pour le decodage
output = output.to("cpu")

# Decoder la sortie en texte lisible
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)