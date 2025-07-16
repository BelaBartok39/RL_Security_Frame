"""
Lightweight monitoring script for production deployment.
Minimal GUI overhead, focuses on anomaly detection performance.
"""

import time
import logging
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import SystemConfig
from processing_pipeline import RFProcessingPipeline


logger = logging.getLogger(__name__)
console = Console()


class LightweightMonitor:
    """Lightweight console-based monitor for RF anomaly detection."""
    
    def __init__(self, pipeline: RFProcessingPipeline):
        self.pipeline = pipeline
        self.start_time = time.time()
        self.last_anomaly = None
        self.anomaly_history = []
        
        # Set callbacks
        pipeline.set_anomaly_callback(self.on_anomaly)
        pipeline.set_metrics_callback(self.on_metrics)
        
        self.current_metrics = {}
    
    def on_anomaly(self, result):
        """Handle anomaly detection."""
        self.last_anomaly = {
            'time': datetime.now(),
            'type': result.classification_result.jammer_type if result.classification_result else 'Unknown',
            'score': result.anomaly_result.anomaly_score,
            'confidence': result.anomaly_result.confidence,
            'action': result.action_taken
        }
        self.anomaly_history.append(self.last_anomaly)
        
        # Console alert
        console.print(f"\n[bold red]⚠️  ANOMALY DETECTED![/bold red]")
        console.print(f"Type: {self.last_anomaly['type']}")
        console.print(f"Score: {self.last_anomaly['score']:.2f}")
        console.print(f"Action: {self.last_anomaly['action']}\n")
    
    def on_metrics(self, metrics):
        """Update metrics."""
        self.current_metrics = metrics
    
    def create_status_table(self):
        """Create status table."""
        table = Table(title="System Status", expand=True)
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Pipeline metrics
        pipeline_metrics = self.current_metrics.get('pipeline', {})
        table.add_row("Packets Processed", str(pipeline_metrics.get('packets_processed', 0)))
        table.add_row("Anomalies Detected", str(pipeline_metrics.get('anomalies_detected', 0)))
        table.add_row("Channel Hops", str(pipeline_metrics.get('channel_hops', 0)))
        table.add_row("Processing Time", f"{pipeline_metrics.get('processing_time_ms', 0):.1f} ms")
        
        # Anomaly metrics
        ad_metrics = self.current_metrics.get('anomaly_detector', {})
        threshold_stats = ad_metrics.get('threshold_stats', {})
        table.add_row("Current Threshold", f"{threshold_stats.get('threshold', 0):.3f}")
        table.add_row("Anomaly Rate", f"{ad_metrics.get('anomaly_rate', 0):.2%}")
        
        # GPU status
        import torch
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            table.add_row("GPU Memory", f"{memory_used:.1f}/{memory_total:.1f} GB")
            table.add_row("GPU Utilization", f"{torch.cuda.utilization()}%")
        
        return table
    
    def create_anomaly_panel(self):
        """Create anomaly history panel."""
        if not self.anomaly_history:
            return Panel("No anomalies detected", title="Anomaly History")
        
        # Show last 5 anomalies
        lines = []
        for anomaly in self.anomaly_history[-5:]:
            lines.append(f"[{anomaly['time'].strftime('%H:%M:%S')}] "
                        f"{anomaly['type']} (score: {anomaly['score']:.2f})")
        
        return Panel("\n".join(lines), title=f"Recent Anomalies ({len(self.anomaly_history)} total)")
    
    def create_channel_panel(self):
        """Create channel status panel."""
        channel_data = self.current_metrics.get('channel_scanner', {})
        current_freq = channel_data.get('current_channel', 0)
        
        lines = [f"Current: {current_freq/1e6:.1f} MHz"]
        lines.append(f"Clean channels: {channel_data.get('clean_channels', 0)}/{channel_data.get('total_channels', 0)}")
        
        if self.last_anomaly and self.last_anomaly.get('action'):
            lines.append(f"\nLast action: {self.last_anomaly['action']}")
        
        return Panel("\n".join(lines), title="Channel Status")
    
    def run(self):
        """Run the lightweight monitor."""
        console.print("[bold green]RF Anomaly Detection Monitor[/bold green]")
        console.print("Lightweight mode - Press Ctrl+C to stop\n")
        
        # Start pipeline
        self.pipeline.start()
        
        try:
            with Live(console=console, refresh_per_second=2) as live:
                while True:
                    # Create layout
                    layout = Layout()
                    layout.split_column(
                        Layout(self.create_status_table(), name="status"),
                        Layout(self.create_anomaly_panel(), name="anomalies"),
                        Layout(self.create_channel_panel(), name="channels")
                    )
                    
                    # Update display
                    live.update(layout)
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            self.pipeline.stop()
            
            # Final summary
            runtime = time.time() - self.start_time
            console.print(f"\n[bold]Session Summary:[/bold]")
            console.print(f"Runtime: {runtime/60:.1f} minutes")
            console.print(f"Total anomalies: {len(self.anomaly_history)}")
            
            if self.current_metrics.get('pipeline'):
                p = self.current_metrics['pipeline']
                console.print(f"Packets processed: {p.get('packets_processed', 0)}")
                console.print(f"Channel hops: {p.get('channel_hops', 0)}")


def main():
    """Main entry point for lightweight monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightweight RF Monitor")
    parser.add_argument('--config', default='config.json', help='Configuration file')
    parser.add_argument('--log-level', default='WARNING', help='Log level')
    
    args = parser.parse_args()
    
    # Minimal logging for production
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = SystemConfig.from_file(args.config)
    
    # Force GPU usage
    config.anomaly_detection.device = 'cuda'
    
    # Create pipeline and monitor
    pipeline = RFProcessingPipeline(config)
    monitor = LightweightMonitor(pipeline)
    
    # Run
    monitor.run()


if __name__ == "__main__":
    main()