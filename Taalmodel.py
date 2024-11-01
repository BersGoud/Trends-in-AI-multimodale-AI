from transformers import AutoModelForCausalLM, AutoTokenizer

class TaalModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def genereer_beschrijving(self, kenmerken):
        if isinstance(kenmerken, list) and all(isinstance(x, float) for x in kenmerken):
            kenmerken = kenmerken[:10]  # Limiteer tot 10 kenmerken voor betere interpretatie

            # Omzetten van de kenmerken naar een tekstuele representatie
            features_description = ", ".join([f"feature {i+1}: {kenmerken[i]:.2f}" for i in range(len(kenmerken))])

            # Maak een duidelijke en beknopte prompt
            prompt = (f"Here are the extracted features from an image: {features_description}. "
                      "Based on these features, describe what the image might depict. "
                      "Focus on the colors, objects, and emotions that might be present in the image. "
                      "Be creative and provide a narrative description.")

            # Coderen van de prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)

            # Pad token ID instellen
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Stel pad token in

            # Attention mask aanmaken
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

            try:
                # Genereer beschrijving met veilige limieten
                gen_tokens = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,  # Verhoog het aantal nieuwe tokens dat moet worden gegenereerd
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.9  # Probeer een iets hogere temperatuur voor creativiteit
                )
                beschrijving = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

                # Post-processen om herhalingen te verminderen
                beschrijving = self.remove_repetitions(beschrijving)

                return beschrijving
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                return None
        else:
            raise ValueError("Kenmerken moeten een lijst van floats zijn.")

    def remove_repetitions(self, text):
        # Verwijder repetitieve zinnen door gebruik te maken van een set
        seen = set()
        output = []
        for line in text.split('\n'):
            if line not in seen:
                seen.add(line)
                output.append(line)
        return '\n'.join(output)