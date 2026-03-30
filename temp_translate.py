from modules.translate_logic import translate

text = "i am a great chef"
direction = "en-hi"
result = translate(text, direction)
print(f"Translated: {result}")
