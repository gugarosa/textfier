from pdfminer.high_level import extract_text


class Extractor:
    """An Extractor class is responsible for extracting and pre-processing text from PDF files.

    """

    def __init__(self, file_path):
        """Initialization method.

        Args:
            file_path (str): Path to the .pdf file to be extracted.
            
        """

        # Extracting .pdf data into a `raw` property
        self.raw = extract_text(file_path)
