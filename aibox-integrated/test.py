import os
x = next(os.walk("flow/"))[2]
y = "flow/" + next(os.walk("flow/"))[2][0]
print(sorted(next(os.walk("flow/"))[2]))