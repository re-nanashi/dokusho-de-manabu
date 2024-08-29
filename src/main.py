import spacy

nlp = spacy.blank("ja")
doc = nlp("今若者たちは1日中生活のあらゆる場面を撮影してそれをシェア共有することが当たり前のことになっています。")
sentences = [sent.text for sent in doc.sents]
print(sentences)
