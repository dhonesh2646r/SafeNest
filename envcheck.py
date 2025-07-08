import google.generativeai as genai

genai.configure(api_key="")

try:
    model = genai.GenerativeModel(model_name="models/gemini-pro")
    response = model.generate_content("Hey Gemini, can you hear me?")
    print("✅ Gemini API Key is working!")
    print("Response:", response.text)
except Exception as e:
    print("❌ API Key failed or not working.")
    print("Error:", str(e))
