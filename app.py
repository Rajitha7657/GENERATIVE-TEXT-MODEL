import streamlit as st
from transformers import pipeline, set_seed

# Set page title
st.set_page_config(page_title="Text Generator using GPT-Neo")

# Title
st.title("📝 Topic-Based Text Generator")
st.subheader("Generate a coherent paragraph on any topic using GPT-Neo")

# Load model
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

generator = load_generator()

# Set random seed for reproducibility
set_seed(42)

# User input
topic = st.text_input("Enter a topic:", placeholder="e.g. climate change, artificial intelligence")

max_len = st.slider("Paragraph length (max tokens)", 50, 300, 150)

# Generate button
if st.button("Generate Paragraph"):
    if topic.strip():
        with st.spinner("Generating..."):
            prompt = f"Write a paragraph about {topic}:"
            result = generator(prompt, max_length=max_len, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=2.0)
            st.success("Done!")
            st.text_area("Generated Paragraph:", result[0]['generated_text'], height=200)
    else:
        st.warning("Please enter a topic.")
