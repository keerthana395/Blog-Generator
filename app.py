import streamlit as st
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM

def getllamaresponse(input_text, no_of_words, blog_style):
    llm = AutoModelForCausalLM.from_pretrained(
    "model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    max_new_tokens=256,
    temperature=0.01
)

    
    template = """
    Write a blog for {blog_style} job profile on the topic '{input_text}' within {no_of_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_of_words"], template=template)
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_of_words=no_of_words))
    
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs", layout="centered", initial_sidebar_state="collapsed")
st.header("Generate Blogs")

input_text = st.text_input("Enter the Blog topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_of_words = st.text_input("No of words")
with col2:
    blog_style = st.selectbox("Writing the blog for", ("Researchers", "Data Scientists", "Common People"), index=0)

submit = st.button("Generate")
if submit:
    st.write(getllamaresponse(input_text, no_of_words, blog_style))
