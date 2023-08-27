import tkinter as tk
from tkinter import ttk

class SimpleCalculator(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Simple Calculator')
        self.geometry("400x400")

        self.createWidgets()

    def createWidgets(self):
        # Entry to display and enter the equation
        self.entry = ttk.Entry(self, font=("Arial", 24))
        self.entry.grid(row=0, column=0, columnspan=4)

        # Buttons
        buttons = [
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            'C', '0', '=', '+'
        ]

        row = 1
        col = 0
        for button in buttons:
            ttk.Button(self, text=button, command=lambda b=button: self.buttonClick(b)).grid(row=row, column=col, sticky='news')
            col += 1
            if col > 3:
                col = 0
                row += 1

        # Adjust column and row weights so they expand appropriatley
        for i in range(4):
            self.grid_columnconfigure(i, weight=1)
            self.grid_rowconfigure(i, weight=1)
        self.grid_rowconfigure(4, weight=1)

    def buttonClick(self, char):
        currentText = self.entry.get()
        if char == 'C':
            self.entry.delete(0, tk.END)
        elif char == '=':
            try:
                result = eval(currentText)
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except Exception as e:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        else:
            self.entry.insert(tk.END, char)

if __name__ == "__main__":
    app = SimpleCalculator()
    app.mainloop()


