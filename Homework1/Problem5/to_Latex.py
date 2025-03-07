

def to_latex(array):
    latex = "$$"
    for i in range(len(array)):
        if i == 0 or array[i] < 0:
            latex+=f"{array[i]}x^{i}"
        else:
            latex+=f"+{array[i]}x^{i}"

    latex+="$$"
    return latex