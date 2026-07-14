import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# STEP 1: LOAD THE GEMINI API KEY
# =========================================================

# Load environment variables from the .env file
load_dotenv()


# Read the GEMINI_API_KEY variable
gemini_api_key = os.getenv("GEMINI_API_KEY")


# Stop the program if the API key is missing
if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Please add GEMINI_API_KEY inside your .env file."
    )


# =========================================================
# TOOL 1: GET COURSE DETAILS
# =========================================================

@tool
def get_course_detail(detail_name: str) -> str:
    """
    Get exact information about the AI course.

    Available details:
    fee, duration, mentor, mode and placement.
    """

    course_data = {
        "fee": "45000",
        "duration": "12 weeks",
        "mentor": "Hassan",
        "mode": "Online",
        "placement": "Placement assistance is available"
    }

    # Clean the input received from the model
    cleaned_name = detail_name.lower().strip()

    # Check whether the requested detail exists
    if cleaned_name in course_data:
        return course_data[cleaned_name]

    return (
        "Detail not found. Available details are "
        "fee, duration, mentor, mode and placement."
    )


# =========================================================
# TOOL 2: ADD PERCENTAGE
# =========================================================

@tool
def add_percentage(amount: float, percentage: float) -> float:
    """
    Add a percentage to an amount.

    Use this tool for GST, tax, markup or percentage increase.
    """

    percentage_amount = amount * percentage / 100

    final_amount = amount + percentage_amount

    return round(final_amount, 2)


# =========================================================
# TOOL 3: CALCULATE DISCOUNT
# =========================================================

@tool
def calculate_discount(
    amount: float,
    discount_percentage: float
) -> float:
    """
    Subtract a discount percentage from an amount.

    Use this tool when the user asks for a discounted price.
    """

    discount_amount = amount * discount_percentage / 100

    final_amount = amount - discount_amount

    return round(final_amount, 2)


# =========================================================
# TOOL 4: CALCULATE INSTALLMENT
# =========================================================

@tool
def calculate_installment(
    total_amount: float,
    number_of_installments: int
) -> float:
    """
    Divide a total amount into equal installments.

    Use this tool when the user asks for monthly,
    weekly or equal payment installments.
    """

    if number_of_installments <= 0:
        return 0.0

    installment_amount = total_amount / number_of_installments

    return round(installment_amount, 2)


# =========================================================
# TOOL 5: CALCULATE TOTAL FEE FOR STUDENTS
# =========================================================

@tool
def calculate_total_fee(
    fee_per_student: float,
    number_of_students: int
) -> float:
    """
    Calculate the total fee for multiple students.

    Multiply the fee per student by the number of students.
    """

    if number_of_students < 0:
        return 0.0

    total_fee = fee_per_student * number_of_students

    return round(total_fee, 2)


# =========================================================
# TOOL 6: CALCULATE AVERAGE MARKS
# =========================================================

@tool
def calculate_average(
    mark1: float,
    mark2: float,
    mark3: float
) -> float:
    """
    Calculate the average of three subject marks.
    """

    total_marks = mark1 + mark2 + mark3

    average_marks = total_marks / 3

    return round(average_marks, 2)


# =========================================================
# TOOL 7: GET STUDENT GRADE
# =========================================================

@tool
def get_grade(score: float) -> str:
    """
    Return the student grade based on a score from 0 to 100.

    Use this tool after calculating or receiving average marks.
    """

    if score < 0 or score > 100:
        return "Invalid score. Score must be between 0 and 100."

    if score >= 90:
        return "A+"

    if score >= 80:
        return "A"

    if score >= 70:
        return "B"

    if score >= 60:
        return "C"

    if score >= 50:
        return "D"

    return "Fail"


# =========================================================
# TOOL 8: CHECK COURSE ELIGIBILITY
# =========================================================

@tool
def check_course_eligibility(
    age: int,
    has_basic_python: bool
) -> str:
    """
    Check whether a student is eligible for the AI course.

    A student must be at least 18 years old and must know
    basic Python.
    """

    if age < 18:
        return "Not eligible because the minimum age is 18."

    if not has_basic_python:
        return (
            "Not eligible yet. Complete basic Python before "
            "joining the course."
        )

    return "The student is eligible for the AI course."


# =========================================================
# TOOL 9: CONVERT WEEKS TO DAYS
# =========================================================

@tool
def convert_weeks_to_days(weeks: float) -> float:
    """
    Convert a number of weeks into days.

    Use this tool when course duration is given in weeks
    but the user wants the duration in days.
    """

    days = weeks * 7

    return round(days, 2)


# =========================================================
# TOOL 10: COUNT WORDS
# =========================================================

@tool
def count_words(text: str) -> int:
    """
    Count the total number of words in a sentence or paragraph.
    """

    words = text.split()

    return len(words)


# =========================================================
# STEP 2: CREATE THE GEMINI MODEL
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    api_key=gemini_api_key
)


# =========================================================
# STEP 3: STORE ALL TOOLS IN A LIST
# =========================================================

tools = [
    get_course_detail,
    add_percentage,
    calculate_discount,
    calculate_installment,
    calculate_total_fee,
    calculate_average,
    get_grade,
    check_course_eligibility,
    convert_weeks_to_days,
    count_words
]


# =========================================================
# STEP 4: CREATE THE AGENT
# =========================================================

agent = create_agent(
    model=model,

    tools=tools,

    system_prompt="""
You are a beginner-friendly AI course and student assistant.

You have access to 10 tools.

Tool usage rules:

1. Use get_course_detail for exact course fee, duration,
   mentor, mode or placement information.

2. Use add_percentage for GST, tax, markup or percentage
   increase calculations.

3. Use calculate_discount when a discount must be
   subtracted from an amount.

4. Use calculate_installment when an amount must be divided
   into equal installments.

5. Use calculate_total_fee when calculating fees for
   multiple students.

6. Use calculate_average to calculate the average of
   three subject marks.

7. Use get_grade to find the grade from a score or average.

8. Use check_course_eligibility to check whether a student
   can join the course.

9. Use convert_weeks_to_days when a duration must be
   converted from weeks to days.

10. Use count_words to count words in text.

Important rules:

- Use tools whenever an exact calculation or stored course
  information is required.

- You may call multiple tools for one user question.

- Use the output of one tool as the input of another tool
  when required.

- Do not guess course information.

- Do not manually perform calculations when a relevant
  calculation tool is available.

- Give the final answer clearly and briefly.
"""
)


# =========================================================
# STEP 5: DISPLAY THE AGENT EXECUTION TRACE
# =========================================================

def print_agent_trace(messages):
    """
    Display the complete execution performed by the agent.

    It shows:
    1. User question
    2. Tool selected by the agent
    3. Input given to the tool
    4. Result returned by the tool
    5. Final response
    """

    print("\n" + "=" * 70)
    print("AGENT EXECUTION TRACE")
    print("=" * 70)

    for message in messages:

        # Get the class name of the message
        message_type = message.__class__.__name__

        # -------------------------------------------------
        # Human message
        # -------------------------------------------------

        if message_type == "HumanMessage":

            print("\nUSER QUESTION:")

            print(message.content)

        # -------------------------------------------------
        # AI message
        # -------------------------------------------------

        elif message_type == "AIMessage":

            # Check whether the AI requested any tools
            if message.tool_calls:

                for tool_call in message.tool_calls:

                    print("\nAGENT DECISION:")

                    print(
                        f"Tool selected: "
                        f"{tool_call['name']}"
                    )

                    print(
                        f"Tool input: "
                        f"{tool_call['args']}"
                    )

            # If there is no tool call, it is normally
            # the final answer
            elif message.content:

                print("\nAGENT RESPONSE:")

                print(message.content)

        # -------------------------------------------------
        # Tool result
        # -------------------------------------------------

        elif message_type == "ToolMessage":

            print("\nTOOL OBSERVATION:")

            print(f"Tool name: {message.name}")

            print(f"Tool result: {message.content}")


# =========================================================
# STEP 6: TAKE INPUT FROM THE USER
# =========================================================

question = input("\nEnter your question: ")


# =========================================================
# STEP 7: RUN THE AGENT
# =========================================================

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    },

    # This prevents the agent from calling tools forever
    config={
        "recursion_limit": 20
    }
)


# =========================================================
# STEP 8: PRINT THE COMPLETE AGENT TRACE
# =========================================================

print_agent_trace(result["messages"])


# =========================================================
# STEP 9: PRINT ONLY THE FINAL ANSWER
# =========================================================

final_message = result["messages"][-1]


print("\n" + "=" * 70)

print("FINAL ANSWER")

print("=" * 70)


print(final_message.content)