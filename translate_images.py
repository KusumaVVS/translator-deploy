import requests
import json

def translate_image(image_path, direction='en-hi'):
    url = 'http://localhost:5000/translate_image'
    files = {'image': open(image_path, 'rb')}
    data = {'direction': direction}

    try:
        response = requests.post(url, files=files, data=data)
        result = response.json()
        return {
            'image': image_path,
            'status': response.status_code,
            'extracted_text': result.get('extracted_text', ''),
            'translated_text': result.get('translated_text', ''),
            'error': result.get('error', '')
        }
    except Exception as e:
        return {
            'image': image_path,
            'status': 'Error',
            'error': str(e)
        }
    finally:
        files['image'].close()

# Translate all four images
images = ['eng1.jpg', 'eng2.png', 'eng3.webp', 'eng4.png']

print("Translating images eng1.jpg, eng2.png, eng3.webp, eng4.png")
print("=" * 60)

for image in images:
    print(f"\nTranslating {image}:")
    result = translate_image(image)
    if result['status'] == 200:
        print(f"  Original Text: {result['extracted_text']}")
        print(f"  Translated Text: {result['translated_text']}")
    else:
        print(f"  Error: {result['error']}")

print("\nTranslation complete!")
