import os
from openai import AzureOpenAI
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Add OpenAI library
import openai

openai.api_key = os.getenv('API_KEY')
openai.api_base =  os.getenv('ENDPOINT')
openai.api_type = 'azure' # Necessary for using the OpenAI library with Azure OpenAI
openai.api_version = '2023-03-15-preview' # Latest / target version of the API

# deployment_name = 'qucoon-ml' # SDK calls this "engine", but naming
#                                            # it "deployment_name" for clarity

# client = AzureOpenAI(
#     api_version = openai.api_version,
#     # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
#     azure_endpoint = openai.api_base,
#     azure_deployment = deployment_name,
# )

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

def read_sample(file_path):
    # Specify the file path
    # file_path = 'sample.txt'

    # Open the file and read its contents into a string
    with open(file_path, 'r') as file:
        file_contents = file.read()

    # Now, file_contents contains the entire text from the file
    return file_contents

def factor_crop_rec(factor, factor_value, factor_normal, crop_name):
    exception_status = "NO"
    try:
        response = client.chat.completions.create(
            temperature=0.1,
            # engine=deployment_name,
            model="gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a professional Rural Agronomists that specializes in soil management, crop production, and the application of scientific methods to improve farming practices."},
                {"role": "user", "content": f"""A farmer wants to plant {crop_name} but has a {factor} level of {factor_value} while the normal level is {factor_normal}. 
                                                He needs to adjust the soil to cover the change of {float(factor_value) - float(factor_normal)}. If there is an increase, recommend ways to reduce but if there is deficiency, recommend ways to increase.
                                                Using these value that the farmer has given and the sample format that will be provided below, generate a recommendation for the farmer to cover this change.
                                                If the change is positive, give practical steps on how to increase the level of {factor}. If the change is negative, give practical steps on how to reduce the level of {factor}.
                                                
                                                Sample recommendation:
                                                {read_sample('app/sample_crop.txt')}"""}
            ]
        )
        return response.choices[0].message.content, exception_status
    except Exception as e:
        exception_status="YES"
        return e,exception_status 

# print(factor_crop_rec("Nitrogen", 30, 40, "Beans")[0])

# prefix = "app/sample_recommendations"
# with open(f"{prefix}/Nitrogen_rice_sample.txt", 'w') as file1:
#     file1.write(factor_crop_rec("Nitrogen", 20, 40, "Rice"))

# with open(f"{prefix}/Phosphorus_rice_sample.txt", 'w') as file2:
#     file2.write(factor_crop_rec("Phosphorus", 20, 40, "Rice"))

# with open(f"{prefix}/Potassium_rice_sample.txt", 'w') as file3:
#     file3.write(factor_crop_rec("Potassium", 20, 40, "Rice"))

def factor_fert_rec(N,P,K,N_level,P_level,K_level,N_normal,P_normal,K_normal,crop_name):
    exception_status = "NO"
    try:
        completion = client.chat.completions.create(
            temperature=0.3,
            # engine=deployment_name,
            # model="gpt-3.5-turbo",
            model="gpt-4o",
            messages = [
                {"role": "system", "content": "You are a professional Rural Agronomists that specializes in soil management, crop production, and the application of scientific methods to improve farming practices."},
                {"role": "user", "content": f"""A farmer takes samples of his soil and gets Nitrogen: {N}, Phosphorus {P} and Potassium {K}. He wishes to plant {crop_name}.
                                            
                                                A sample recommendation format is provided below when Nitrogen level is high, Phosphorus normal and potassium low.
                                                Sample recommendation:
                                                {read_sample('app/sample_fert.txt')}
                                                
                                                For this particular farmer, the Nitrogen Level of the soil is {N_level},
                                                the current Phosphorus Level of the soil is {P_level} when compared to a normal value of {P_normal},
                                                Potassium Level of the soil is {K_level}.
                                                
                                                For this soil profile, using the sample format only, provide a recommendation for the soil to effect {N_level} level Nitrogen, {P_level} level Phosphorus and {K_level} level Potassium but nothing more."""}
                ]

            )
        return completion.choices[0].message.content, exception_status
    except Exception as e:
        exception_status="YES"
        return e, exception_status

def factor_disease_rec(crop_name,disease_name):
    exception_status = "NO"
    try:
        completion = client.chat.completions.create(
            temperature=0.1,
            # engine=deployment_name,
            # model="gpt-3.5-turbo",
            model="gpt-4o",
            messages = [
                {"role": "system", "content": "You are a recommender for rural farmers. only use block of text in 100 words strictly and make responses dynamic, The response should be in 2 paragraphs, first paragraph should speak to Causes of the disease and the second paragraph should speak to remedies, you should add kenyan local remedies, if you add Kenyan traditional name, describe it briefly)"},
                {"role": "user", "content": f"""I have been provided with a crop name and disease it is suffering from. In two sections (Cause, Remedies) with a maximum of two bullet points, generate a recommendation for a case of {crop_name} suffering {disease_name}.
                                                
                                                Below is a sample recommendation for tomato suffering from Tomato Late Blight:
                                                {read_sample('app/sample_disease.txt')}"""},
            ]

        )
        return completion.choices[0].message.content, exception_status
    except Exception as e:
        exception_status="YES"
        return e, exception_status
    
# print(factor_fert_rec(82,60,50,"Normal","High","Normal",80,42,50,"Rice")[0])

# print(factor_disease_rec("Apple","Black rot")[0])