import re

def parse_math_question(question):
    """
    Parses a math question by extracting only the part before the equals sign
    and trimming whitespace.
    
    Args:
        question (str): The math question to parse
        
    Returns:
        str: The trimmed expression part (before the equals sign)
    """
    # Split the question at the equals sign and take the first part
    if '=' in question:
        expression_part = question.split('=')[0]
    else:
        expression_part = question
    
    # Trim whitespace
    expression_part = expression_part.strip()
    
    return expression_part

def clean_expression(expression):
    """
    Removes LaTeX formatting and cleans up a math expression
    for computation.
    
    Args:
        expression (str): The expression to clean
        
    Returns:
        str: The cleaned expression
    """
    # Remove dollar signs (LaTeX math delimiters)
    expression = re.sub(r'\$', '', expression)
    
    # Replace LaTeX division symbol
    expression = re.sub(r'\\div', '/', expression)
    
    # Replace LaTeX multiplication symbol
    expression = re.sub(r'\\cdot', '*', expression)
    expression = re.sub(r'\\times', '*', expression)
    
    # Clean up spacing
    expression = re.sub(r'\s+', ' ', expression).strip()
    
    return expression

if __name__ == "__main__":
    # Example usage
    test_questions = [
        "6 ÷ 2(1+2) = ?",
        "2x + 3 = 15",
        "What is 5 + 7?",
        "$6 \\div 2 \\cdot 3 = $"
    ]
    
    for question in test_questions:
        parsed = parse_math_question(question)
        print(f"Original: {question}")
        print(f"Parsed:   {parsed}")
        
        if '\\' in question or '$' in question:
            cleaned = clean_expression(parsed)
            print(f"Cleaned:  {cleaned}")
        
        print() 