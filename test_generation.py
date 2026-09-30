from generate_answer import generate_answer


question = "What should I check if my printer is making a strange noise?"

context = """
Printer Model X Troubleshooting Guide

If the printer is making a strange noise, first check whether
the paper feed mechanism is blocked.

Turn off the printer and inspect the paper feed area for any
stuck paper or foreign objects.

If the problem continues after removing any obstruction,
contact technical support.
"""


answer = generate_answer(question, context)

print("\n===== GENERATED ANSWER =====")
print(answer)