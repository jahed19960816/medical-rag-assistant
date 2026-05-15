from ocr_utils import extract_text_from_scanned_pdf

text = extract_text_from_scanned_pdf("sample.pdf")

print(text[:1000])