import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Verifier si CUDA est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger la configuration
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Specifier le chemin local vers le repertoire du modele (pas le fichier .bin directement)
chemin_local = "Dabococo/owai_model"

# Charger le modele fine-tune a partir du chemin local et le deplacer sur le GPU
model = AutoModelForCausalLM.from_pretrained(chemin_local, config=config).to(device)

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Définir la proportion de la mémoire GPU allouée à ce processus (0.8 pour 80%)
torch.cuda.set_per_process_memory_fraction(0.8)

while True:
    # Poser une nouvelle question a l'utilisateur
    input_text = input("Posez votre question (ou tapez 'exit' pour vous casser d'ici) : ")

    # Verifier si l'utilisateur souhaite quitter
    if input_text.lower() == 'exit':
        print("Au revoir !")
        break

    # Tokeniser le texte d'entree et le deplacer sur le GPU
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generer du texte sur le GPU
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Deplacer la sortie sur le CPU pour le decodage
    output = output.to("cuda")

    # Decoder la sortie en texte lisible
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("OWAI Copilot:", decoded_output)
