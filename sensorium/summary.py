
import os
import sys

result = open(sys.argv[1], "r")

value = 0.0
for line in result:
    value += float(line.split(": ")[1])
print(f"{(value/5):.4f}")

result.close()

