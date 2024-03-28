import os
import json
import pandas as pd
import traceback

import streamlit as st

from dotenv import load_dotenv
from ATSgenerator.ats.atsapp import generate_evaluate_chain
from ATSgenerator.ats.utill import read_file
#from langchain_community.callbacks import get_openai_callback
from langchain.callbacks import get_openai_callback

load_dotenv()

# import warnings
# warnings.filterwarnings('ignore')


st.title(" ATS Tracking System")

with st.form("user_submits"):
    upload_file=st.file_uploader("Upload you resume:(eg-.pdf)")

    job_description=st.text_input("Job description",placeholder="Write you job description")

    button=st.form_submit_button("Write Review about my resume..")



    if button and upload_file is not None:
        with st.spinner("Loading.."):
            try:
               candidate_resume=read_file(upload_file)
               print(candidate_resume)
               response=generate_evaluate_chain(         
                   {
                           "job_description":job_description,
                            "candidate_resume":candidate_resume
                       }
                   )
               st.write(response.get("job_skill"))
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            # else:
            #     print(f"Total Tokens:{cb.total_tokens}")
            #     print(f"Prompt Tokens:{cb.prompt_tokens}")
            #     print(f"Completion Token:{cb.completion_tokens}")
            #     print(f"Total Cost:{cb.total_cost}")







