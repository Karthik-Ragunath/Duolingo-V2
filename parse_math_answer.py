import re

def extract_math_answer(text):
    """
    Remove LaTeX formatting and special symbols from mathematical text
    
    Args:
        text (str): Text containing LaTeX math symbols
        
    Returns:
        str: Cleaned text with special symbols replaced or removed
    """
    # Remove dollar sign delimiters (LaTeX math mode)
    text = re.sub(r'\$', '', text)
    
    # Replace LaTeX symbols with plain text equivalents
    replacements = {
        r'\\div': '/',           # Division symbol
        r'\\cdot': '*',          # Multiplication dot
        r'\\times': '*',         # Multiplication x
        r'\\frac{([^}]+)}{([^}]+)}': r'\1/\2',  # Fractions like \frac{a}{b} to a/b
        r'\\sqrt{([^}]+)}': r'sqrt(\1)',        # Square root
        r'\\sqrt\[([^]]+)\]{([^}]+)}': r'\1-root(\2)',  # nth root
        r'\\left\(': '(',        # Left parenthesis
        r'\\right\)': ')',       # Right parenthesis
        r'\\left\[': '[',        # Left bracket
        r'\\right\]': ']',       # Right bracket
        r'\\infty': 'infinity',  # Infinity symbol
        r'\\pi': 'pi',           # Pi symbol
        r'\\approx': '≈',        # Approximately equal
        r'\\neq': '≠',           # Not equal
        r'\\leq': '≤',           # Less than or equal
        r'\\geq': '≥',           # Greater than or equal
        r'\\boxed{([^}]+)}': r'\1'  # Remove boxed content formatting
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Process superscripts (^) for powers
    text = re.sub(r'(\w+)\^(\w+)', r'\1^\2', text)  # Preserve powers for clarity
    
    # Remove any remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    return text