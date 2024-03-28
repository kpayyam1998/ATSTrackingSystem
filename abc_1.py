import os
import json
import pandas as pd
import traceback
import streamlit as st

#Imported logics
from ATSgenerator.ats.atsapp import generate_evaluate_chain
from ATSgenerator.ats.utill import read_file
from langchain.callbacks import get_openai_callback



st.set_page_config(page_title="Automated Tracking System")
st.title("ATS Tracking System")

with st.form("user_submits"):
    upload_file=st.file_uploader("Upload you resume:",type=["pdf"])

    job_description=st.text_area("Job description",placeholder="Write your job description")

    button=st.form_submit_button("Write Review about my resume..")



    if button and upload_file is not None and job_description :
        with st.spinner("Loading.."):
            try:
               candidate_resume=read_file(upload_file)
               print(candidate_resume)
               with get_openai_callback() as cb:
                    response=generate_evaluate_chain(         
                        {
                           "job_description":job_description,
                            "candidate_resume":candidate_resume
                        })
                        
               st.write(response.get("job_skill"))
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Token:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")







