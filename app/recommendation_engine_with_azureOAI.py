import os
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

# Add OpenAI library
import openai

openai.api_key = os.getenv('API_KEY')
openai.api_base =  os.getenv('ENDPOINT')
openai.api_type = 'azure' # Necessary for using the OpenAI library with Azure OpenAI
openai.api_version = '2023-03-15-preview' # Latest / target version of the API

deployment_name = 'qucoon-ml' # SDK calls this "engine", but naming
                                           # it "deployment_name" for clarity

client = AzureOpenAI(
    api_version=openai.api_version,
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=openai.api_base,
    azure_deployment=deployment_name,
)

def read_sample():
    # Specify the file path
    file_path = 'app/sample.txt'

    # Open the file and read its contents into a string
    with open(file_path, 'r') as file:
        file_contents = file.read()

    # Now, file_contents contains the entire text from the file
    return file_contents

def model(prompt):

    response = client.chat.completions.create(
        temperature=0.1,
        # engine=deployment_name,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a freindly assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# print(model("Hi, what is the fasteset bird in the world?"))


def factor_crop_rec(factor, factor_value, factor_normal, crop_name):
    exception_status = "NO"

    response = client.chat.completions.create(
        temperature=0.1,
        # engine=deployment_name,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional Rural Agronomists that specializes in soil management, crop production, and the application of scientific methods to improve farming practices."},
            {"role": "user", "content": f"""A farmer wants to plant {crop_name} but has a {factor} level of {factor_value} while the normal level is {factor_normal}. 
                                            He needs to adjust the soil to cover the deficit of {float(factor_value) - float(factor_normal)}.
                                            Using these value that the farmer has given and the sample format that will be provided below, generate a recommendation for the farmer to cover this deficit.
                                            
                                            Sample recommendation:
                                            {read_sample()}"""}
        ]
    )
    return response.choices[0].message.content

print(factor_crop_rec("Phosphorus", 20, 40, "Rice"))
