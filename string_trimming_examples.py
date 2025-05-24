"""
String Trimming Examples in Python

This file demonstrates various methods to trim strings in Python,
from basic to advanced techniques.
"""

def basic_trimming():
    """Basic string trimming methods in Python"""
    print("1. Basic String Trimming Methods")
    print("-" * 40)
    
    # Example string with leading and trailing whitespace
    example = "   Hello, World!   "
    print(f"Original string: '{example}'")
    
    # strip() - removes whitespace from both ends
    stripped = example.strip()
    print(f"strip():        '{stripped}'")
    
    # lstrip() - removes whitespace from the left (beginning)
    left_stripped = example.lstrip()
    print(f"lstrip():       '{left_stripped}'")
    
    # rstrip() - removes whitespace from the right (end)
    right_stripped = example.rstrip()
    print(f"rstrip():       '{right_stripped}'")
    
    print()
    
def trimming_specific_characters():
    """Trimming specific characters instead of just whitespace"""
    print("2. Trimming Specific Characters")
    print("-" * 40)
    
    # Example with non-whitespace characters to trim
    example = "###Python Programming!!!"
    print(f"Original string: '{example}'")
    
    # Remove specific characters from both ends
    stripped = example.strip('#!')
    print(f"strip('#!'):    '{stripped}'")
    
    # Remove specific characters from left only
    left_stripped = example.lstrip('#')
    print(f"lstrip('#'):    '{left_stripped}'")
    
    # Remove specific characters from right only
    right_stripped = example.rstrip('!')
    print(f"rstrip('!'):    '{right_stripped}'")
    
    print()

def advanced_trimming():
    """Advanced string trimming techniques"""
    print("3. Advanced Trimming Techniques")
    print("-" * 40)
    
    import re
    
    # Example string with various patterns
    example = "   <-- Python Programming -->   "
    print(f"Original string: '{example}'")
    
    # Using regular expressions for more complex trimming
    # Remove whitespace and arrow patterns from both ends
    regex_stripped = re.sub(r'^[\s<-]+|[\s>-]+$', '', example)
    print(f"regex trim:     '{regex_stripped}'")
    
    # Using slicing to trim fixed number of characters
    slice_trimmed = example[5:-5]
    print(f"slice[5:-5]:    '{slice_trimmed}'")
    
    print()

def real_world_examples():
    """Real-world examples of string trimming"""
    print("4. Real-World Trimming Examples")
    print("-" * 40)
    
    # CSV data cleaning
    csv_entry = "  42, Python,  Data Science  "
    cleaned_csv = [field.strip() for field in csv_entry.split(',')]
    print(f"CSV Original:  '{csv_entry}'")
    print(f"CSV Cleaned:   {cleaned_csv}")
    
    # URL path cleaning
    url_path = "/api/users/123/"
    trimmed_path = url_path.strip('/')
    print(f"URL Original:  '{url_path}'")
    print(f"URL Trimmed:   '{trimmed_path}'")
    
    # Math expression cleaning
    math_expr = "6 ÷ 2(1+2) = ?"
    expr_part = math_expr.split('=')[0].strip()
    print(f"Math Original: '{math_expr}'")
    print(f"Expression:    '{expr_part}'")
    
    print()

if __name__ == "__main__":
    print("String Trimming in Python\n")
    basic_trimming()
    trimming_specific_characters()
    advanced_trimming()
    real_world_examples() 