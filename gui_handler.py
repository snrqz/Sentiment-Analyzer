from tkinter import *
from sentiments_main import *
from preprocessing_txt import *

clf = obtain_model()
root = Tk()
sentiments = ["sad", "joy", "love", "anger", "fear", "surprise"]
root.geometry(f"750x750+{(root.winfo_screenwidth() // 2) - (750 // 2)}+{(root.winfo_screenheight() // 2) - (750 // 2)}")


def obtain_text():
    if clf is None:
        raise ValueError("Can't predict, classifier is NULL")
    else:
        text = text_insertion.get('1.0', END).strip()
        preprocessed_text = pre_process_sentence(text)
        lemmatized_text = lemmatize_text(preprocessed_text)
        text_output.config(text=f"Prediction: {sentiments[(clf.predict([lemmatized_text]))[0]]}")


text_insertion = Text(root, height=10, width=15, font=("Arial Black", 25))
text_insertion.pack(expand=True, fill="both", padx=10, pady=10)


button = Button(root,
                   text="Click Me", command=obtain_text,
                   activebackground="green", activeforeground="white",
                   font=("Arial Black", 25)
                )

button.pack(pady=10)

text_output = Label(root, font=("Arial Black", 25), text="")
text_output.pack(pady=10)

root.mainloop()
