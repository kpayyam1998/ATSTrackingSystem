import os
import traceback 
import pandas as pd
import json


from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback



from dotenv import load_dotenv


import warnings 
warnings.filterwarnings('ignore')

load_dotenv()


key=os.getenv("OPENAI_API_KEY")


#model
llm=ChatOpenAI(openai_api_key=key,model="gpt-3.5-turbo-0613",temperature=0.5)

# Candidate Template
Template_1="""
Here i have provoided the candidate resume.
you role and responsibility is to find out the below task

1. Find out the how many years of experience needed (e.g : fresher,experinece or in numbers 1,2 3,etc
2. Find out the specific tool if they mentioned any specific tool experienced person
3. Find out if any certificate releated to the job description
4. Find out the skill,programming languages 
5. Find out other information related to job description
6. Find out the candidate qualification
7. Find out the Key Responsibilities:
8. Find out the Requirements:

Here below the resume information
##########################
{candidate_resume}
"""

Resume_template=PromptTemplate(
    input_variables=["candidate_resume"],
    template=Template_1
)

resume_chain=LLMChain(llm=llm,prompt=Resume_template,output_key="candidate_info",verbose=True)

# Job Description Template

Template="""
Here below i have given the candidate resume information
{candidate_info}

Here i have given the Job description
###################################
{job_description}
Your are the profile matchin expertise.please match these resume and job description 

Perform below task
***********************
1. Find out the how many years of experience needed (e.g : fresher,experinece or in numbers 1,2 3,etc
2. Find out the specific tool if they mentioned any specific tool experienced person
3. Find out if any certificate releated to the job description
4. Find out the skill,programming languages.
6. Find out the candidate qualification 
7. Find out the Key Responsibilities:
8. Find out the Requirements:

Can you tell me the matching percentage of these ?
Can you please match candidate profile to jobdescription
Can you please suggest what need to be imporve



"""
Job_template=PromptTemplate(
    input_variables=["candidate_info","job_description"],
    template=Template

)

job_chain=LLMChain(llm=llm,prompt=Job_template,output_key="job_skill",verbose=True)



# Sequential Chain
generate_evaluate_chain=SequentialChain(chains=[resume_chain,job_chain],
                input_variables=["job_description","candidate_resume"],
                output_variables=["candidate_info","job_skill"])









