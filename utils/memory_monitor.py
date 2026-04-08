# utils/memory_monitor.py
"""
Lightweight GPU memory monitor.
"""

import torch
import time
import threading
import logging
from typing import Dict, List


class MemoryMonitor:
    """GPU memory monitor."""

    def __init__(self, interval: float = 5.0):
        """
        Args:
            interval: polling interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.logger = logging.getLogger("memory_monitor")
        self.stats = {
            "max_allocated": 0,
            "max_reserved": 0,
            "allocated_history": [],
            "reserved_history": [],
            "timestamps": []
        }

    def start(self):
        """Start monitoring in a background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("Memory monitoring started")

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.logger.info("Memory monitoring stopped")

    def _monitor_loop(self):
        """Polling loop."""
        start_time = time.time()

        while self.monitoring:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3

                self.stats["max_allocated"] = max(self.stats["max_allocated"], allocated)
                self.stats["max_reserved"] = max(self.stats["max_reserved"], reserved)

                self.stats["allocated_history"].append(allocated)
                self.stats["reserved_history"].append(reserved)
                self.stats["timestamps"].append(time.time() - start_time)

                # Threshold logs (tuned for large GPUs; adjust if needed)
                if allocated > 40:
                    self.logger.warning(f"High GPU memory usage: {allocated:.2f} GB")
                elif allocated > 35:
                    self.logger.info(f"GPU memory usage: {allocated:.2f} GB / {reserved:.2f} GB")

            time.sleep(self.interval)

    def get_report(self) -> Dict:
        """Return a summary dict of collected stats."""
        if not self.stats["allocated_history"]:
            return {"error": "No data collected"}

        return {
            "peak_allocated_gb": self.stats["max_allocated"],
            "peak_reserved_gb": self.stats["max_reserved"],
            "avg_allocated_gb": sum(self.stats["allocated_history"]) / len(self.stats["allocated_history"]),
            "avg_reserved_gb": sum(self.stats["reserved_history"]) / len(self.stats["reserved_history"]),
            "monitoring_duration": self.stats["timestamps"][-1] if self.stats["timestamps"] else 0,
            "sample_count": len(self.stats["allocated_history"])
        }

    def print_summary(self):
        """Print a human-readable summary."""
        report = self.get_report()

        print("\n" + "=" * 50)
        print("GPU memory usage summary")
        print("=" * 50)
        print(f"Peak allocated: {report['peak_allocated_gb']:.2f} GB")
        print(f"Peak reserved: {report['peak_reserved_gb']:.2f} GB")
        print(f"Avg allocated: {report['avg_allocated_gb']:.2f} GB")
        print(f"Duration: {report['monitoring_duration']:.0f} s")
        print("=" * 50)