import zipfile

with zipfile.ZipFile("processed.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
    