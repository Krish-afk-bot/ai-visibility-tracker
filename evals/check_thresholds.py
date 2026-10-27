import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.thresholds import load_thresholds

def check_thresholds(offline_path: str = None, analyzer_path: str = None) -> bool:
    thresholds = load_thresholds()
    passed = True
    
    offline_data = None
    if offline_path and Path(offline_path).exists():
        with open(offline_path, "r", encoding="utf-8") as f:
            offline_data = json.load(f)
            
    analyzer_data = None
    if analyzer_path and Path(analyzer_path).exists():
        with open(analyzer_path, "r", encoding="utf-8") as f:
            analyzer_data = json.load(f)
            
    print("=" * 60)
    print("EVALUATION THRESHOLD VERIFICATION")
    print("=" * 60)
    
    # 1. Check Mention F1
    if "mention_f1" in thresholds:
        thresh = thresholds["mention_f1"]
        val = None
        if offline_data and "mention_detection" in offline_data:
            val = offline_data["mention_detection"].get("f1")
        if val is not None:
            ok = val >= thresh
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] mention_f1: {val:.4f} (Threshold: >={thresh:.4f})")
            if not ok:
                passed = False
        else:
            print(f"[SKIP] mention_f1: no evaluation result found (Threshold: >={thresh:.4f})")

    # 2. Check Position within-1 accuracy
    if "position_within1" in thresholds:
        thresh = thresholds["position_within1"]
        val = None
        if offline_data and "position_metrics" in offline_data:
            val = offline_data["position_metrics"].get("accuracy_within_1")
        if val is not None:
            ok = val >= thresh
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] position_within1: {val:.4f} (Threshold: >={thresh:.4f})")
            if not ok:
                passed = False
        else:
            print(f"[SKIP] position_within1: no evaluation result found (Threshold: >={thresh:.4f})")

    # 3. Check Competitor Fallback F1
    if "competitor_fallback_f1" in thresholds:
        thresh = thresholds["competitor_fallback_f1"]
        val = None
        if offline_data and "competitor_extraction" in offline_data:
            val = offline_data["competitor_extraction"].get("f1")
        if val is not None:
            ok = val >= thresh
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] competitor_fallback_f1: {val:.4f} (Threshold: >={thresh:.4f})")
            if not ok:
                passed = False
        else:
            print(f"[SKIP] competitor_fallback_f1: no evaluation result found (Threshold: >={thresh:.4f})")

    # 4. Check Sentiment Accuracy (if analyzer results provided)
    if "sentiment_accuracy" in thresholds:
        thresh = thresholds["sentiment_accuracy"]
        val = None
        if analyzer_data and "sentiment" in analyzer_data:
            val = analyzer_data["sentiment"].get("accuracy")
        if val is not None:
            ok = val >= thresh
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] sentiment_accuracy: {val:.4f} (Threshold: >={thresh:.4f})")
            if not ok:
                passed = False
        else:
            print(f"[INFO] sentiment_accuracy: skipped (live analyzer eval not provided)")

    print("-" * 60)
    if passed:
        print("ALL THRESHOLDS SATISFIED.")
    else:
        print("FAILURES DETECTED: One or more evaluation metrics fell below thresholds.")
    print("=" * 60)
    
    return passed

def main():
    parser = argparse.ArgumentParser(description="Check evaluation results against minimum thresholds.")
    parser.add_argument("--offline", type=str, default="evals/results/offline_results.json", help="Path to offline results JSON")
    parser.add_argument("--analyzer", type=str, default=None, help="Path to analyzer results JSON (optional)")
    args = parser.parse_args()

    ok = check_thresholds(offline_path=args.offline, analyzer_path=args.analyzer)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
