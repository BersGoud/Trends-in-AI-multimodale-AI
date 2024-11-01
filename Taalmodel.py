# taalmodel.py
from transformers import AutoModelForCausalLM, AutoTokenizer

class TaalModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    def genereer_beschrijving(self, kenmerken):
        if isinstance(kenmerken, list) and all(isinstance(x, float) for x in kenmerken):
            # Beperk tot de eerste 100 kenmerken of een andere geschikte limiet
            kenmerken = kenmerken[:100]
            
            # Omzetten naar tekst
            kenmerken_tekst = ", ".join(map(str, kenmerken))

            # Coderen van de tekst met truncatie
            input_ids = self.tokenizer.encode(kenmerken_tekst, return_tensors='pt', max_length=1024, truncation=True)
            print("Tokenized Input ID Length:", input_ids.shape[1])  # Controleer de lengte

            # Instellen van de pad-token en attention mask
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Zorg ervoor dat de pad-token correct is ingesteld
            attention_mask = (input_ids != self.tokenizer.pad_token_id).int()  # Maak een tensor van de attention mask
            
            # Debugging info
            print("Input IDs:", input_ids)
            print("Attention Mask:", attention_mask)

            # Genereer beschrijving
            try:
                # Hier beperkt max_new_tokens
                gen_tokens = self.model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    max_length=1024 + 50,  # Totaal 1074 tokens
                    num_return_sequences=1, 
                    do_sample=True, 
                    temperature=0.7
                )
                beschrijving = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                return beschrijving
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                return None
        else:
            raise ValueError("Kenmerken moeten een lijst van floats zijn.")