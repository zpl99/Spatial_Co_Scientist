import os
from pdfminer.high_level import extract_text


def pdf_to_txt(pdf_path, txt_path):
    text = extract_text(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)


def batch_convert_pdf_to_txt(pdf_dir, txt_dir):

    os.makedirs(txt_dir, exist_ok=True)


    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(txt_dir, txt_filename)

            print(f"Converting: {pdf_path} --> {txt_path}")
            try:
                pdf_to_txt(pdf_path, txt_path)
            except Exception as e:
                print(f"Failed to convert {pdf_path}: {e}")


if __name__ == '__main__':
    pdf_dir = "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/data/spatial_cluster_paper_pdf"
    txt_dir = "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/data/spatial_cluster_paper_txt"
    batch_convert_pdf_to_txt(pdf_dir, txt_dir)
