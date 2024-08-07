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

print(model("Hi, what is the fasteset bird in the world?"))