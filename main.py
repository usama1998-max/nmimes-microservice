# main.py

import json
import os
import re

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from typing import Optional
import google.generativeai as genai
from settings import Settings
# import sympy
# from sympy import symbols, Eq, solve, pi, sin, cos, tan, simplify, nsimplify
# from sympy.parsing.sympy_parser import (
#     parse_expr,
#     standard_transformations,
#     implicit_multiplication_application
# )
# import re
# from typing import Dict
import uvicorn
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
gemini_client_pro = genai.GenerativeModel("gemini-2.5-pro-preview-03-25")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] to be specific
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = openai.OpenAI(
    api_key=Settings.OPENAI_API_KEY
)

deep_seek = openai.OpenAI(
    api_key=Settings.DEEPSEEK_API_KEY,
    base_url=Settings.DEEPSEEK_BASE_URL
)


class ChatRequest(BaseModel):
    prompt: Optional[str]
    history: Optional[str]


class AnalysisRequest(BaseModel):
    history: Optional[str]

# transformations = standard_transformations + (implicit_multiplication_application,)

# def validate_math_question(data: Dict):
#     question = data["question"]
#     options = data["options"]
#     correct_option = data["correctOption"]
#
#     try:
#         # Evaluate the expression safely
#         parsed = parse_expr(question)
#         result = float(parsed.evalf())
#
#         # Round if necessary for comparison
#         rounded_result = round(result, 1) if isinstance(correct_option, float) else round(result)
#
#         is_correct = rounded_result == correct_option
#
#         return {
#             "valid": is_correct,
#             "evaluated_answer": rounded_result,
#             "message": "✅ Valid" if is_correct else f"❌ Incorrect — should be {rounded_result}",
#             "question": question,
#             "options": options
#         }
#
#     except Exception as e:
#         return {
#             "valid": False,
#             "evaluated_answer": None,
#             "message": f"Error parsing question: {e}",
#             "question": question,
#             "options": options
#         }


# def parse_and_eval_expr(expr_str: str):
#     expr_str = expr_str.replace("°", "*pi/180")  # handle degrees
#     expr = parse_expr(expr_str.strip(), transformations=transformations)
#     return expr.evalf()
#
#
# def validate_math_question(data: Dict):
#     question = data["question"].strip()
#     options = data["options"]
#     correct_option = data["correctOption"]
#
#     try:
#         # Handling algebraic equations like "3x + 5 = 20"
#         if '=' in question:
#             lhs, rhs = question.split('=')
#             lhs_expr = parse_expr(lhs.split(':')[-1].strip(), transformations=transformations)
#             rhs_expr = parse_expr(rhs.strip(), transformations=transformations)
#             eq = Eq(lhs_expr, rhs_expr)
#             sol = solve(eq)
#
#             if not sol:
#                 return {
#                     "valid": False,
#                     "evaluated_answer": None,
#                     "message": "❌ No solution found.",
#                     "question": question,
#                     "options": options
#                 }
#
#             result = sol[0]
#             topic = "Algebra"
#
#         # Handling percentage-based questions like "40% of 50"
#         elif '%' in question:
#             # Improved regex to extract percentage and total number
#             match = re.search(r'(\d+)\s*%.*(\d+)', question)
#             if match:
#                 pct_value = float(match.group(1))  # 40%
#                 total_value = float(match.group(2))  # 50
#                 result = (pct_value / 100) * total_value
#                 topic = "Percentage"
#             else:
#                 raise ValueError("Percentage or total number could not be extracted.")
#
#         # Handling ratios and proportions like "3:5 = 15:x"
#         elif ':' in question:
#             match = re.search(r'(\d+):(\d+)\s*=\s*(\d+):x', question)
#             if match:
#                 ratio1 = float(match.group(1))
#                 ratio2 = float(match.group(2))
#                 value = float(match.group(3))
#                 result = (value * ratio2) / ratio1
#                 topic = "Ratio/Proportion"
#             else:
#                 raise ValueError("Ratio or proportion format is incorrect.")
#
#         # Handling general arithmetic questions like "5 + 10", "12 - 3", "6 * 4"
#         else:
#             expr_str = question.split(':')[-1].strip()
#             result = parse_and_eval_expr(expr_str)
#             topic = "Arithmetic"
#
#         # Normalize result and correct_option
#         rounded_result = round(float(result), 2)
#         rounded_correct = round(float(correct_option), 2)
#         is_correct = rounded_result == rounded_correct
#
#         return {
#             "valid": is_correct,
#             "evaluated_answer": rounded_result,
#             "message": "✅ Valid" if is_correct else f"❌ Incorrect — should be {rounded_result}",
#             "question": question,
#             "options": options,
#             "topic": topic
#         }
#
#     except Exception as e:
#         return {
#             "valid": False,
#             "evaluated_answer": None,
#             "message": f"⚠️ Error parsing or solving question: {e}",
#             "question": question,
#             "options": options
#         }


async def generate_question(prompt: str, history: str):
    response = llm.chat.completions.create(
        model="gpt-4o-mini",

        messages=[
            {"role": "system", "content": """### Be a question generator, Create easy, medium or hard version of question depending on the user previous response. If a user finds maths OK, return hard question, if a user finds math Great return easy or medium question, if a user finds math boring return easy question. return JSON format same as Previous User Response. Only modify the question key. [/INST] 

                   ## Things to Avoid:
                   - Description
                   - Bullet Points
                   - Extra response with json
                   - Whitespaces
                   - ASCII characters
                   - Description with question
                   - Including topic with the question and response
                   - Regenerating same question and its values
                   - Same 4 options
                   - We cannot use float for nouns like there are 14.5 boys etc
                   - whole text question with values
                   

                   ## Things to Focus: 
                   - Relative question generation depending on user response
                   - Provide options as well as correct option
                   - Options should always be list format with integers and no SI unit or any other thing
                   - Always add correct option
                   - Make sure is from the topic provided by user
                   - Only the equation should be provided
                   - Make sure if decimal or float is provided, do add that as well
                   - Descriptive question can be same with different values inside
                   - All options should be different and 1 of them should be correct
                   - convert any ASCII character for division, multiplication, addition, subtraction, percentage
                   - extract main equation rather than whole question
                   - question should be same but values different
                   
                    
                   ## Respond in JSON! 
                   ### RESPONSE FORMAT:
                   {
                       "question": "",
                       "options": [provide 4 max options, should not contain any SI unit just integer or float],
                       "correctOption": ""
                   }
                   
                   """ + f"""## Previous Responses: {history}\n\n"""},
            {"role": "user", "content":  prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content


def prompt_format(prompt: str, history: str):
    return """Be a question generator, Create easy, medium or hard version of question depending on the user previous response. If a user finds maths OK, return hard question, if a user finds math Great return easy or medium question, if a user finds math boring return easy question. return JSON format same as Previous User Response. Only modify the question key.

   Things to avoid:
   - Description
   - Bullet Points
   - Extra response with json
   - Whitespaces
   - ASCII characters
   - Description with question
   - Including topic with the question and response
   - Regenerating same question and its values
   - Same 4 options
   - We cannot use float for nouns like there are 14.5 boys etc
   - whole text question with values
   - ```json``` tags or any code related tags.
   

   Things to focus on: 
   - Relative question generation depending on user response
   - Provide options as well as correct option
   - Options should always be list format with integers and no SI unit or any other thing
   - Always add correct option
   - Make sure is from the topic provided by user
   - Only the equation should be provided
   - Make sure if decimal or float is provided, do add that as well
   - Descriptive question can be same with different values inside
   - All options should be different and 1 of them should be correct
   - convert any ASCII character for division, multiplication, addition, subtraction, percentage
   - extract main equation rather than whole question
   - Only JSON response without ```json ``` format.
   
    
   Respond in JSON! 
   RESPONSE FORMAT:
   {
       "question": "",
       "options": [provide 4 max options, should not contain any SI unit just integer or float],
       "correctOption": ""
   }
""" + f"{history} \n\n{prompt}"


def analyze_prompt(history: str):
    response = llm.chat.completions.create(
        model="gpt-4o-mini",

        messages=[
            {"role": "user", "content":"""
            Question Number 1
            Topic: Emotional check-In
            Question: How do you feel about mathematics?
            User Answered: Difficult
            Correct Answer:
            Time Taken: 2025-05-24 23:55
            
            
            Question Number 2
            Topic: Basic Arithmetic
            Question: Solve the equation: 18 / 3 + (7 * 2)
            User Answered: 22
            Correct Answer: 24
            Time Taken: 2025-05-24 23:55
            
            
            Question Number 3
            Topic: Fractions & Decimals
            Question: What is the total amount of liquid if I have 3/2 of coconut water and 0.25 of juice?
            User Answered: 2.0
            Correct Answer: 1.75
            Time Taken: 2025-05-24 23:55
            
            
            Question Number 4
            Topic: Simple Equations
            Question: Find x : 4x - 6 = 10
            User Answered: 7
            Correct Answer: 4
            Time Taken: 2025-05-24 23:55
            
            
            
            Question Number 5
            Topic: Percentages & Ratios
            Question: I collected 30 precious stones. If exactly 40% are emeralds, how many are NOT emeralds?
            User Answered: 15
            Correct Answer: 18
            Time Taken: 2025-05-24 23:56
            
            
            
            Question Number 6
            Topic: Geometry
            Question: If a triangle has one angle of 90 degrees and another angle of 45 degrees, find the third angle.
            User Answered: 135
            Correct Answer: 45
            Time Taken: 2025-05-24 23:56
            
            
            
            Question Number 7
            Topic: Algebraic Expressions
            Question: If 5x - 7 = 3x + 11, what is the value of x?
            User Answered: 7
            Correct Answer: 9
            Time Taken: 2025-05-24 23:56
            
            
            
            Question Number 8
            Topic: Word Problems
            Question: Solve the equation: 15 / 5 + (6 * 2)
            User Answered: 19
            Correct Answer: 17
            Time Taken: 2025-05-24 23:57
            
            
            Question Number 9
            Topic: Logical reasoning
            Question: What is the value of x in the equation 3x + 5 = 20?
            User Answered: 8
            Correct Answer: 5            
            Time Taken: 2025-05-24 23:59

            ### Based on the given user assessment, provide the level (1, 2 or 3) and create a lesson plan or roadmap for user based on week. Provide a json response. 
            ### Keep the topic name same, do not create topic name from yourself. Select same topic name from the history of user. 
            ## Follow the format described below:

            {
            
            "level": ".."
            
            "task": [
            
             {
                'week': '...',
                'tasks': [
                  {'done': false, 'task': '...'},
                ],
            },
            
            ]
            "weakness": "...",
            "strength": "..."
            }"""},
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content


async def regenerate_question(prompt: str):
    response = llm.chat.completions.create(
        model="gpt-4o-mini", # o4, o3, o1

        messages=[
            {"role": "system", "content": """### Correct the following question. Make sure to answer is correct.

                       ## Things to Avoid:
                       - Description
                       - Bullet Points
                       - Extra response with json
                       - whitespaces
                       - ASCII characters
                       - description with question
                       - including topic with the question and response
                       - new options
                       
                       

                       ## Things to focus on: 
                       - Provide options as well as correct option
                       - Options should always be list format with integers or float and no SI unit or any other text or description
                       - Always add correct option
                       - Make sure is from the topic provided by user
                       - Only the equation should be provided
                       - Make sure if decimal or float is provided, do add that as well
                       - Must include correct option
                       - Question should make sense
                       - Make answer is correct and is provided in the option
                       

                       ## Respond in JSON! 
                       RESPONSE FORMAT:
                       {
                           "question": "",
                           "options": [provide 4 max options, should not contain any SI unit just integer or float],
                           "correctOption": ""
                       }               
                       """},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content


@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        print(request.history, request.prompt)
        gq = await generate_question(request.prompt, request.history)
        rgq = await regenerate_question(gq)
        rrgq = await regenerate_question(rgq)

        cleaned = re.sub(r"^```json\s*|\s*```$", "", rrgq.strip())
        print(gq)
        print(rgq)
        print(rrgq)

        return JSONResponse({"response": json.loads(cleaned)},
                            status_code=status.HTTP_200_OK,
                            media_type="application/json; charset=utf-8")
    except Exception as e:
        print(e)
        return JSONResponse({"error": "Error during execution of chat!"},
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            media_type="application/json; charset=utf-8")


@app.post("/chat/gemini/")
async def chat_gemini(request: ChatRequest):
    print(request)
    try:
        response = gemini_client_pro.generate_content(prompt_format(request.prompt, request.history), generation_config={"temperature": 0.7})
        cleaned = re.sub(r"^```json\s*|\s*```$", "", response.text.strip())

        # Step 2: Load as JSON
        data = json.loads(cleaned)
        print(data)
        return JSONResponse({"response": data},
                            status_code=status.HTTP_200_OK,
                            media_type="application/json; charset=utf-8")

    except Exception as e:

        logger.error(str(e))

        # Optional: customize message based on error type content
        error_message = str(e).lower()
        if "rate limit" in error_message:
            return JSONResponse({"error": "Rate limit exceeded!"},
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                media_type="application/json; charset=utf-8")

        elif "unauthorized" in error_message or "403" in error_message:
            return JSONResponse({"error": "Unauthorized or invalid API key!"},
                                status_code=status.HTTP_403_FORBIDDEN,
                                media_type="application/json; charset=utf-8")

        elif "quota" in error_message:
            return JSONResponse({"error": "Quota exceeded!"},
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                media_type="application/json; charset=utf-8")

        elif any(keyword in error_message for keyword in [
            "generation", "content", "invalid prompt", "model", "safety"
        ]):
            return JSONResponse({"error": "Cannot generate response due to generation error!"},
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                media_type="application/json; charset=utf-8")

        else:
            return JSONResponse({"error": "Error during execution of chat!"},
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            media_type="application/json; charset=utf-8")


@app.post("/analyze/")
async def analyze_assessment(request: AnalysisRequest):
    print(request)
    try:
        response = analyze_prompt(request.history)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())

        # Step 2: Load as JSON
        data = json.loads(cleaned)
        print(data)
        return JSONResponse({"response": data},
                            status_code=status.HTTP_200_OK,
                            media_type="application/json")

    except Exception as e:

        logger.error(str(e))

        # Optional: customize message based on error type content
        error_message = str(e).lower()
        if "rate limit" in error_message:
            return JSONResponse({"error": "Rate limit exceeded!"},
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS)
        elif "unauthorized" in error_message or "403" in error_message:
            return JSONResponse({"error": "Unauthorized or invalid API key!"},
                                status_code=status.HTTP_403_FORBIDDEN)
        elif "quota" in error_message:
            return JSONResponse({"error": "Quota exceeded!"},
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS)
        elif any(keyword in error_message for keyword in [
            "generation", "content", "invalid prompt", "model", "safety"
        ]):
            return JSONResponse({"error": "Cannot generate response due to generation error!"},
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return JSONResponse({"error": "Error during execution of chat!"},
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
