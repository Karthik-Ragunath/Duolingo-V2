#!/usr/bin/env python3
"""
Test script for the math answer parsing functionality.
This script tests various formats of math answers to ensure our parsing is robust.
"""

from parse_math_answer import extract_math_answer

def test_parsing_functions():
    """Test the parsing functionality with various answer formats"""
    test_cases = [
        # Test case 1: Answer with "The answer is: $X$" format
        {
            "input": """First, we need to solve the expression inside the parentheses: $1+2=3$.
So the equation becomes $6 \\div 2 \\cdot 3$.
Next, we perform the multiplication and division from left to right: $6 \\div 2 = 3$, then $3 \\cdot 3 = 9$.
So the answer is $9$. The answer is: $9$""",
            "expected": "9",
            "description": "Basic 'The answer is: $X$' format"
        },
        
        # Test case 2: Answer with \boxed{} format
        {
            "input": """To solve this problem, I'll follow PEMDAS.
First, I calculate what's in the parentheses: $(2+3) = 5$
Then I multiply: $4 \\times 5 = 20$
Finally, I add: $7 + 20 = 27$
Therefore, the answer is \\boxed{27}""",
            "expected": "27",
            "description": "\\boxed{} format"
        },
        
        # Test case 3: Answer with "The answer is X" format (no dollar signs)
        {
            "input": """To find the area of the rectangle, I multiply length by width.
Length = 8 cm, Width = 5 cm
Area = 8 × 5 = 40
The answer is 40 square centimeters.""",
            "expected": "40 square centimeters",
            "description": "'The answer is X' format (no dollar signs)"
        },
        
        # Test case 4: Complex answer in boxed format
        {
            "input": """To solve this system of equations:
$2x + 3y = 8$
$4x - y = 7$

I'll solve for x in terms of y from the second equation:
$4x - y = 7$
$4x = 7 + y$
$x = \\frac{7 + y}{4}$

Now I'll substitute this into the first equation:
$2(\\frac{7 + y}{4}) + 3y = 8$
$\\frac{2(7 + y)}{4} + 3y = 8$
$\\frac{14 + 2y}{4} + 3y = 8$
$14 + 2y + 12y = 32$
$14 + 14y = 32$
$14y = 18$
$y = \\frac{18}{14} = \\frac{9}{7}$

Now I can find x:
$x = \\frac{7 + \\frac{9}{7}}{4} = \\frac{49 + 9}{28} = \\frac{58}{28} = \\frac{29}{14}$

Therefore, the solution is \\boxed{x = \\frac{29}{14}, y = \\frac{9}{7}}""",
            "expected": "x = \\frac{29}{14}, y = \\frac{9}{7}",
            "description": "Complex answer in boxed format"
        },
        
        # Test case 5: Multiple boxed answers (should take the last one)
        {
            "input": """Step 1: Find the derivative of f(x) = x^2 + 3x.
f'(x) = 2x + 3
Step 2: Evaluate at x = 4.
f'(4) = 2(4) + 3 = 8 + 3 = 11
The derivative is \\boxed{11}

Alternatively, we can verify our answer.
f(x+h) - f(x) / h as h approaches 0
= ((x+h)^2 + 3(x+h) - (x^2 + 3x)) / h
= (x^2 + 2xh + h^2 + 3x + 3h - x^2 - 3x) / h
= (2xh + h^2 + 3h) / h
= 2x + h + 3
As h approaches 0, we get 2x + 3
At x = 4, we get 2(4) + 3 = 11
So the answer is \\boxed{f'(4) = 11}""",
            "expected": "f'(4) = 11",
            "description": "Multiple boxed answers (should take the last one)"
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        result = extract_math_answer(test_case["input"])
        
        if result == test_case["expected"]:
            print(f"✅ Test {i} passed: {test_case['description']}")
            success_count += 1
        else:
            print(f"❌ Test {i} failed: {test_case['description']}")
            print(f"   Expected: {test_case['expected']}")
            print(f"   Actual: {result}")
    
    print(f"\nSummary: {success_count}/{len(test_cases)} tests passed")

if __name__ == "__main__":
    test_parsing_functions() 