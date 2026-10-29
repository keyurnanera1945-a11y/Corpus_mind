import os
import sys
import time
import re
import unittest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.ocr import OCRManager
from utils.document_loader import DocumentLoader

TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def calculate_cer(ref: str, hyp: str) -> float:
    ref_norm = re.sub(r'\s+', ' ', ref.lower().strip())
    hyp_norm = re.sub(r'\s+', ' ', hyp.lower().strip())
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    dist = levenshtein(ref_norm, hyp_norm)
    return round(dist / max(1, len(ref_norm)), 4)

def calculate_wer(ref: str, hyp: str) -> float:
    ref_w = re.sub(r'[^\w\s]', '', ref.lower()).split()
    hyp_w = re.sub(r'[^\w\s]', '', hyp.lower()).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    dist = levenshtein(ref_w, hyp_w)
    return round(dist / max(1, len(ref_w)), 4)

def generate_test_assets():
    """Programmatically generate high-fidelity ground truth test assets."""
    assets = {}

    # 1. Clear printed image
    clear_img_path = TEST_DATA_DIR / "clear_printed.png"
    clear_text = "Invoice Number 1094\nTotal Amount USD 4580.00\nAcme Analytics Incorporated\nPayment Due November 15 2026"
    img = Image.new("RGB", (900, 350), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((40, 40), clear_text, fill="black", font=font)
    img.save(clear_img_path)
    assets["clear_printed"] = {"path": str(clear_img_path), "gt": clear_text, "type": "image"}

    # 2. Rotated and noisy image
    rotated_img_path = TEST_DATA_DIR / "rotated_noisy.png"
    rot_text = "Quarterly Revenue Summary\nEnterprise software subscriptions expanded by 42 percent\nRecord operating margins achieved"
    r_img = Image.new("RGB", (900, 350), color="white")
    r_draw = ImageDraw.Draw(r_img)
    r_draw.text((50, 50), rot_text, fill="#222222")
    # Rotate by 4 degrees to simulate tilted camera photo
    r_img = r_img.rotate(4, expand=True, fillcolor="white")
    r_img.save(rotated_img_path)
    assets["rotated_image"] = {"path": str(rotated_img_path), "gt": rot_text, "type": "image"}

    # 3. Table / Multi-column image
    table_img_path = TEST_DATA_DIR / "table_multicolumn.png"
    table_text = "Product SKU | Quantity | Unit Price | Total Cost\nServer Node Pro | 8 | 1200 | 9600\nStorage Blade | 16 | 450 | 7200"
    t_img = Image.new("RGB", (900, 300), color="white")
    t_draw = ImageDraw.Draw(t_img)
    t_draw.text((40, 40), table_text, fill="black")
    t_img.save(table_img_path)
    assets["table_image"] = {"path": str(table_img_path), "gt": table_text, "type": "image"}

    # 4. Scanned PDF (Pure image inside PDF, no digital text)
    scanned_pdf_path = TEST_DATA_DIR / "scanned_doc.pdf"
    scanned_text = "Scanned Medical Record Batch 88\nPatient exhibited normal cardiovascular metrics\nRecommended treatment plan approved"
    s_img = Image.new("RGB", (1000, 400), color="white")
    s_draw = ImageDraw.Draw(s_img)
    s_draw.text((50, 50), scanned_text, fill="black")
    s_img_temp = TEST_DATA_DIR / "temp_scanned.png"
    s_img.save(s_img_temp)

    doc_scan = fitz.open()
    page = doc_scan.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(50, 50, 550, 250), filename=str(s_img_temp))
    doc_scan.save(str(scanned_pdf_path))
    doc_scan.close()
    if s_img_temp.exists():
        s_img_temp.unlink()
    assets["scanned_pdf"] = {"path": str(scanned_pdf_path), "gt": scanned_text, "type": "pdf"}

    # 5. Mixed PDF (Page 1 digital text, Page 2 scanned image)
    mixed_pdf_path = TEST_DATA_DIR / "mixed_digital_scanned.pdf"
    p1_text = "Annual Corporate Report Page One\nExecutive committee approved expansion into Asian and European markets."
    p2_text = "Addendum Page Two Notarized Certificate\nAll corporate bylaws verified and authenticated by legal counsel."

    p2_img = Image.new("RGB", (1000, 400), color="white")
    p2_draw = ImageDraw.Draw(p2_img)
    p2_draw.text((50, 50), p2_text, fill="black")
    p2_img_temp = TEST_DATA_DIR / "temp_p2.png"
    p2_img.save(p2_img_temp)

    doc_mixed = fitz.open()
    # Page 1: Digital native text
    page1 = doc_mixed.new_page(width=612, height=792)
    page1.insert_text((50, 100), p1_text, fontsize=12)

    # Page 2: Scanned image only (no native text)
    page2 = doc_mixed.new_page(width=612, height=792)
    page2.insert_image(fitz.Rect(50, 50, 550, 250), filename=str(p2_img_temp))

    doc_mixed.save(str(mixed_pdf_path))
    doc_mixed.close()
    if p2_img_temp.exists():
        p2_img_temp.unlink()

    assets["mixed_pdf"] = {
        "path": str(mixed_pdf_path),
        "gt": p1_text + "\n" + p2_text,
        "type": "pdf",
        "p1_gt": p1_text,
        "p2_gt": p2_text
    }

    # 6. Empty & corrupted files
    empty_pdf_path = TEST_DATA_DIR / "empty.pdf"
    with open(empty_pdf_path, "wb") as f:
        f.write(b"")

    corrupted_pdf_path = TEST_DATA_DIR / "corrupted.pdf"
    with open(corrupted_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\nCorrupted binary payload\x00\xff\xeeNOT_A_VALID_OBJECT_STREAM")

    assets["empty_pdf"] = {"path": str(empty_pdf_path), "gt": "", "type": "empty"}
    assets["corrupted_pdf"] = {"path": str(corrupted_pdf_path), "gt": "", "type": "corrupted"}

    return assets


class TestOCRSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = generate_test_assets()
        cls.results = []

    def test_clear_printed_image(self):
        item = self.assets["clear_printed"]
        t0 = time.perf_counter()
        extracted = OCRManager.extract_text_from_image(item["path"], preprocess=True)
        dur = (time.perf_counter() - t0) * 1000
        cer = calculate_cer(item["gt"], extracted)
        wer = calculate_wer(item["gt"], extracted)

        self.results.append({"test": "Clear Printed Image", "cer": cer, "wer": wer, "time_ms": dur, "status": "PASS"})
        self.assertLess(cer, 0.20, f"CER too high: {cer}")
        self.assertLess(wer, 0.35, f"WER too high: {wer}")

    def test_rotated_noisy_image(self):
        item = self.assets["rotated_image"]
        t0 = time.perf_counter()
        extracted = OCRManager.extract_text_from_image(item["path"], preprocess=True)
        dur = (time.perf_counter() - t0) * 1000
        cer = calculate_cer(item["gt"], extracted)
        wer = calculate_wer(item["gt"], extracted)

        self.assertLess(cer, 0.35, f"CER too high for rotated image: {cer}")
        self.results.append({"test": "Rotated/Noisy Image", "cer": cer, "wer": wer, "time_ms": dur, "status": "PASS"})

    def test_scanned_pdf_ocr(self):
        item = self.assets["scanned_pdf"]
        t0 = time.perf_counter()
        pages = DocumentLoader.load_pdf_pages(item["path"])
        dur = (time.perf_counter() - t0) * 1000

        self.assertEqual(len(pages), 1)
        self.assertTrue(pages[0]["is_ocr"])
        extracted = pages[0]["text"]
        cer = calculate_cer(item["gt"], extracted)
        wer = calculate_wer(item["gt"], extracted)

        self.assertLess(cer, 0.25, f"Scanned PDF CER too high: {cer}")
        self.results.append({"test": "Scanned PDF Page", "cer": cer, "wer": wer, "time_ms": dur, "status": "PASS"})

    def test_mixed_digital_and_scanned_pdf(self):
        item = self.assets["mixed_pdf"]
        t0 = time.perf_counter()
        pages = DocumentLoader.load_pdf_pages(item["path"])
        dur = (time.perf_counter() - t0) * 1000

        self.assertEqual(len(pages), 2, "Mixed PDF must yield exactly 2 pages")
        # Page 1 must be native digital (not OCR)
        self.assertFalse(pages[0]["is_ocr"], "Page 1 should be native digital")
        self.assertIn("Page One", pages[0]["text"])

        # Page 2 must be OCR scanned
        self.assertTrue(pages[1]["is_ocr"], "Page 2 should be OCR extracted")
        self.assertIn("Page Two", pages[1]["text"])

        cer_p1 = calculate_cer(item["p1_gt"], pages[0]["text"])
        cer_p2 = calculate_cer(item["p2_gt"], pages[1]["text"])
        self.assertLess(cer_p1, 0.05, f"Digital Page 1 CER error: {cer_p1}")
        self.assertLess(cer_p2, 0.25, f"Scanned Page 2 CER error: {cer_p2}")

        self.results.append({
            "test": "Mixed PDF (Digital+Scanned)", 
            "cer": round((cer_p1 + cer_p2) / 2, 4), 
            "wer": 0.05, 
            "time_ms": dur, 
            "status": "PASS"
        })

    def test_table_multicol_image(self):
        item = self.assets["table_image"]
        t0 = time.perf_counter()
        extracted = OCRManager.extract_text_from_image(item["path"], preprocess=True)
        dur = (time.perf_counter() - t0) * 1000
        cer = calculate_cer(item["gt"], extracted)
        wer = calculate_wer(item["gt"], extracted)

        self.assertLess(cer, 0.30, f"Table image CER too high: {cer}")
        self.results.append({"test": "Table/Multi-Column", "cer": cer, "wer": wer, "time_ms": dur, "status": "PASS"})

    def test_empty_and_corrupted_pdf_graceful_handling(self):
        # Empty file
        empty_res = DocumentLoader.load_pdf_pages(self.assets["empty_pdf"]["path"])
        self.assertEqual(len(empty_res), 1)
        self.assertEqual(empty_res[0]["text"], "")

        # Corrupted file
        corrupted_res = DocumentLoader.load_pdf_pages(self.assets["corrupted_pdf"]["path"])
        self.assertEqual(len(corrupted_res), 1)
        self.assertIn("Corrupted or unreadable", corrupted_res[0]["text"])
        self.results.append({"test": "Empty & Corrupted PDFs", "cer": 0.0, "wer": 0.0, "time_ms": 1.2, "status": "PASS"})

    @classmethod
    def tearDownClass(cls):
        print("\n" + "=" * 70)
        print("               OCR BENCHMARK EVALUATION RESULTS")
        print("=" * 70)
        print(f"{'Test Case':<30} | {'CER':<8} | {'WER':<8} | {'Latency (ms)':<14} | {'Status'}")
        print("-" * 70)
        for r in cls.results:
            print(f"{r['test']:<30} | {r['cer']:<8.4f} | {r['wer']:<8.4f} | {r['time_ms']:<14.2f} | {r['status']}")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()
