
'''
text = "🙃"
print (text)
text = text.encode("utf-8")
print (text)
text = text.decode("utf-8")
print (text)
'''

text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
print (text)
text = text.encode("utf-8")
print (text)
text = text.decode("utf-8")
print (text)
