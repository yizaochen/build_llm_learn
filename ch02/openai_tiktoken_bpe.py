import tiktoken

print(f"tiktoken.__version__: {tiktoken.__version__}")

tokenizer = tiktoken.get_encoding("gpt2")

# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#      "of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))