import os
import json
def files_to_documents():
    files = []
    for file in os.listdir("./corpus"):
        with open(os.path.join("./corpus", file), "r", encoding='utf-8') as infile:
            files.append(infile.read())


    with open("out.json", 'w') as outfile:
        json.dump(files, outfile)

files_to_documents()
    