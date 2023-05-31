import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to the Titanic ML Explorer! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar**  to see what are some of the common steps you can apply in a ML pipeline.

     ## How it works:
     In this demo you will apply each of the steps in our ML pipeline with the push of a button!

     ## Try it yourself!
     In this demo you will use the model trained by giving it your own set of parameters.

"""
)