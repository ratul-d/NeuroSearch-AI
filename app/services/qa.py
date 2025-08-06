import os
import groq

def ask_llama_3_70b(query, retrieved_chunks):

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")
    client = groq.Client(api_key=api_key)
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content