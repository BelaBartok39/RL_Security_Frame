"""
Real-time monitoring GUI for RF anomaly detection system.
Uses tkinter for compatibility and simplicity on Jetson.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime
import matplotlib

from processing_pipeline import RFProcessingPipeline
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from collections import deque


logger = logging.getLogger(__name__)


class RFMonitoringGUI:
    """
    Real-time monitoring GUI for the RF security framework.
    """
    
    def __init__(self, pipeline: Optional['RFProcessingPipeline'] = None):
        """
        Initialize GUI.
        
        Args:
            pipeline: Processing pipeline to monitor
        """
        self.pipeline = pipeline
        self.root = tk.Tk()
        self.root.title("RF Anomaly Detection Monitor")
        self.root.geometry("1400x900")
        
        # Data queues for thread-safe updates
        self.anomaly_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        
        # Data storage for plots
        self.anomaly_scores = deque(maxlen=300)
        self.timestamps = deque(maxlen=300)
        self.threshold_values = deque(maxlen=300)
        
        # Signal visualization data
        self.signal_queue = queue.Queue(maxsize=10)
        self.spectrogram_data = deque(maxlen=100)  # For waterfall display
        self.constellation_points = {'i': deque(maxlen=1000), 'q': deque(maxlen=1000)}
        
        # Setup GUI components
        self._setup_gui()
        
        # Connect to pipeline if provided
        if pipeline:
            self.connect_pipeline(pipeline)
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _setup_gui(self):
        """Setup GUI components."""
        # Create main frames
        self.create_menu_bar()
        
        # Top frame for status
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        self.create_status_panel()
        
        # Middle frame with notebook for different views
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_realtime_tab()
        self.create_signal_tab()
        self.create_anomaly_tab()
        self.create_channel_tab()
        self.create_log_tab()
    
    def create_menu_bar(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Models", command=self.save_models)
        file_menu.add_command(label="Load Models", command=self.load_models)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Control menu
        control_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Control", menu=control_menu)
        control_menu.add_command(label="Start Pipeline", command=self.start_pipeline)
        control_menu.add_command(label="Stop Pipeline", command=self.stop_pipeline)
        control_menu.add_separator()
        control_menu.add_command(label="Reset Threshold", command=self.reset_threshold)
        control_menu.add_command(label="Channel Sweep", command=self.channel_sweep)
    
    def create_status_panel(self):
        """Create status panel with key metrics."""
        # Pipeline status
        self.status_label = ttk.Label(self.status_frame, text="Status: Disconnected", 
                                     font=('Arial', 12, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Key metrics
        self.metrics_frame = ttk.Frame(self.status_frame)
        self.metrics_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.metric_labels = {}
        metrics = ['Packets', 'Anomalies', 'Channel Hops', 'Processing (ms)']
        for i, metric in enumerate(metrics):
            label = ttk.Label(self.metrics_frame, text=f"{metric}: --")
            label.grid(row=0, column=i, padx=20)
            self.metric_labels[metric] = label
    
    def create_realtime_tab(self):
        """Create real-time monitoring tab."""
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="Real-time Monitor")
        
        # Create matplotlib figure for anomaly scores
        self.fig = Figure(figsize=(12, 6), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (samples)')
        self.ax.set_ylabel('Anomaly Score')
        self.ax.set_title('Real-time Anomaly Detection')
        self.ax.grid(True)
        
        # Plot lines (will be updated)
        self.anomaly_line, = self.ax.plot([], [], 'b-', label='Anomaly Score')
        self.threshold_line, = self.ax.plot([], [], 'r--', label='Threshold')
        self.ax.legend()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=realtime_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(realtime_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Window:").pack(side=tk.LEFT, padx=5)
        self.window_var = tk.StringVar(value="300")
        window_combo = ttk.Combobox(control_frame, textvariable=self.window_var,
                                   values=["100", "300", "500", "1000"],
                                   width=10)
        window_combo.pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Clear", command=self.clear_plot).pack(side=tk.LEFT, padx=20)
    
    def create_signal_tab(self):
        """Create signal visualization tab with spectrogram and constellation."""
        signal_frame = ttk.Frame(self.notebook)
        self.notebook.add(signal_frame, text="Signal Visualization")
        
        # Create figure with subplots
        self.signal_fig = Figure(figsize=(14, 8), dpi=80)
        
        # Spectrogram subplot
        self.spec_ax = self.signal_fig.add_subplot(221)
        self.spec_ax.set_title('Spectrogram (FFT Magnitude)')
        self.spec_ax.set_xlabel('Frequency Bin')
        self.spec_ax.set_ylabel('Time')
        
        # Initialize spectrogram image
        self.spec_image = self.spec_ax.imshow(
            np.zeros((100, 512)), 
            aspect='auto', 
            cmap='viridis',
            interpolation='nearest',
            vmin=0, 
            vmax=1
        )
        self.signal_fig.colorbar(self.spec_image, ax=self.spec_ax, label='Magnitude')
        
        # Constellation plot
        self.const_ax = self.signal_fig.add_subplot(222)
        self.const_ax.set_title('I/Q Constellation')
        self.const_ax.set_xlabel('In-phase (I)')
        self.const_ax.set_ylabel('Quadrature (Q)')
        self.const_ax.grid(True, alpha=0.3)
        self.const_ax.set_xlim(-3, 3)
        self.const_ax.set_ylim(-3, 3)
        
        # Initialize constellation scatter
        self.const_scatter = self.const_ax.scatter([], [], c='blue', alpha=0.5, s=1)
        
        # Time domain plot
        self.time_ax = self.signal_fig.add_subplot(223)
        self.time_ax.set_title('Time Domain Signal')
        self.time_ax.set_xlabel('Sample')
        self.time_ax.set_ylabel('Amplitude')
        self.time_ax.grid(True, alpha=0.3)
        
        # Initialize time domain lines
        self.i_line, = self.time_ax.plot([], [], 'b-', label='I', alpha=0.7)
        self.q_line, = self.time_ax.plot([], [], 'r-', label='Q', alpha=0.7)
        self.time_ax.legend()
        
        # Power spectrum plot
        self.psd_ax = self.signal_fig.add_subplot(224)
        self.psd_ax.set_title('Power Spectral Density')
        self.psd_ax.set_xlabel('Frequency Bin')
        self.psd_ax.set_ylabel('Power (dB)')
        self.psd_ax.grid(True, alpha=0.3)
        
        # Initialize PSD line
        self.psd_line, = self.psd_ax.plot([], [], 'g-')
        
        # Embed in tkinter
        self.signal_canvas = FigureCanvasTkAgg(self.signal_fig, master=signal_frame)
        self.signal_canvas.draw()
        self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(signal_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Update Rate:").pack(side=tk.LEFT, padx=5)
        self.signal_update_var = tk.StringVar(value="10")
        update_combo = ttk.Combobox(control_frame, textvariable=self.signal_update_var,
                                   values=["1", "5", "10", "30"],
                                   width=10)
        update_combo.pack(side=tk.LEFT)
        ttk.Label(control_frame, text="Hz").pack(side=tk.LEFT, padx=5)
        
        self.pause_signal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Pause", 
                       variable=self.pause_signal_var).pack(side=tk.LEFT, padx=20)
        
        # Test signal button
        ttk.Button(control_frame, text="Test Signal", 
                  command=self.generate_test_signal).pack(side=tk.LEFT, padx=20)
    
    def create_anomaly_tab(self):
        """Create anomaly detection details tab."""
        anomaly_frame = ttk.Frame(self.notebook)
        self.notebook.add(anomaly_frame, text="Anomaly Details")
        
        # Split into two parts
        left_frame = ttk.Frame(anomaly_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(anomaly_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Anomaly list
        ttk.Label(left_frame, text="Recent Anomalies:", font=('Arial', 10, 'bold')).pack()
        
        # Treeview for anomaly list
        columns = ('Time', 'Type', 'Score', 'Confidence', 'Action')
        self.anomaly_tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.anomaly_tree.heading(col, text=col)
            self.anomaly_tree.column(col, width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.anomaly_tree.yview)
        self.anomaly_tree.configure(yscrollcommand=scrollbar.set)
        
        self.anomaly_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Statistics panel
        ttk.Label(right_frame, text="Detection Statistics:", font=('Arial', 10, 'bold')).pack(pady=10)
        
        self.stats_text = tk.Text(right_frame, width=40, height=20, wrap=tk.WORD)
        self.stats_text.pack(padx=10, pady=5)
    
    def create_channel_tab(self):
        """Create channel status tab."""
        channel_frame = ttk.Frame(self.notebook)
        self.notebook.add(channel_frame, text="Channel Status")
        
        # Channel list
        ttk.Label(channel_frame, text="Channel Status:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        # Create channel status display
        columns = ('Frequency', 'Energy (dBm)', 'Status', 'Last Scan', 'Jammed')
        self.channel_tree = ttk.Treeview(channel_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.channel_tree.heading(col, text=col)
            self.channel_tree.column(col, width=120)
        
        self.channel_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Hop history
        ttk.Label(channel_frame, text="Channel Hop History:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.hop_text = scrolledtext.ScrolledText(channel_frame, height=10, wrap=tk.WORD)
        self.hop_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def create_log_tab(self):
        """Create log tab."""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="System Log")
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(log_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=5)
    
    def connect_pipeline(self, pipeline):
        """Connect to processing pipeline."""
        self.pipeline = pipeline
        
        # Set callbacks
        pipeline.set_anomaly_callback(self.on_anomaly_detected)
        pipeline.set_metrics_callback(self.on_metrics_update)
        pipeline.set_signal_callback(self.on_signal_data)
        
        self.status_label.config(text="Status: Connected")
        self.log_message("Connected to processing pipeline")
    
    def on_anomaly_detected(self, result):
        """Callback for anomaly detection."""
        # Queue for thread-safe update
        self.anomaly_queue.put(result)
    
    def on_metrics_update(self, metrics):
        """Callback for metrics update."""
        # Queue for thread-safe update
        self.metrics_queue.put(metrics)
    
    def on_signal_data(self, signal_data):
        """Callback for signal visualization data."""
        # Log signal data reception at INFO level so we can see it
        logger.info(f"GUI received signal data: type={type(signal_data)}, keys={list(signal_data.keys()) if isinstance(signal_data, dict) else 'N/A'}")
        
        # Queue for thread-safe update
        try:
            self.signal_queue.put_nowait(signal_data)
            logger.info(f"Signal data queued successfully, queue size: {self.signal_queue.qsize()}")
        except queue.Full:
            logger.warning("Signal queue is full, dropping data")
            pass  # Drop if queue is full
    
    def _update_loop(self):
        """Background thread for updating GUI."""
        last_debug_time = time.time()
        last_signal_update = time.time()
        
        while self.running:
            try:
                # Process anomaly queue
                anomaly_count = 0
                while not self.anomaly_queue.empty():
                    result = self.anomaly_queue.get_nowait()
                    self._process_anomaly(result)
                    anomaly_count += 1
                
                # Process metrics queue
                metrics_count = 0
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    self._update_metrics(metrics)
                    metrics_count += 1
                
                # Process signal queue at configured rate
                if not self.pause_signal_var.get():
                    update_rate = float(self.signal_update_var.get())
                    if time.time() - last_signal_update > (1.0 / update_rate):
                        signal_processed = 0
                        while not self.signal_queue.empty():
                            signal_data = self.signal_queue.get_nowait()
                            self._process_signal(signal_data)
                            signal_processed += 1
                        
                        if signal_processed > 0:
                            logger.info(f"Processing {signal_processed} signal frames, updating plots")
                            self._update_signal_plots()
                        
                        last_signal_update = time.time()
                
                # Debug output every 5 seconds
                if time.time() - last_debug_time > 5:
                    if anomaly_count > 0 or metrics_count > 0:
                        logger.info(f"GUI processed {anomaly_count} anomalies, {metrics_count} metrics")
                    last_debug_time = time.time()
                
                # Update plots
                self._update_plot()
                
                time.sleep(0.1)  # 10 Hz update rate
                
            except Exception as e:
                logger.error(f"GUI update error: {e}")
                import traceback
                traceback.print_exc()
    
    def _process_anomaly(self, result):
        """Process anomaly result for display."""
        # Add to plot data
        self.timestamps.append(len(self.timestamps))
        self.anomaly_scores.append(result.anomaly_result.anomaly_score)
        self.threshold_values.append(result.anomaly_result.threshold)
        
        # Add to anomaly list if detected
        if result.anomaly_result.is_anomaly:
            time_str = datetime.fromtimestamp(result.timestamp).strftime('%H:%M:%S')
            jammer_type = result.classification_result.jammer_type if result.classification_result else "Unknown"
            score = f"{result.anomaly_result.anomaly_score:.2f}"
            confidence = f"{result.anomaly_result.confidence:.2f}"
            action = result.action_taken or "None"
            
            # Insert at top of tree
            self.anomaly_tree.insert('', 0, values=(time_str, jammer_type, score, confidence, action))
            
            # Keep list bounded
            if len(self.anomaly_tree.get_children()) > 100:
                self.anomaly_tree.delete(self.anomaly_tree.get_children()[-1])
            
            # Log message
            self.log_message(f"ANOMALY: {jammer_type} detected, score={score}, action={action}")
    
    def _update_metrics(self, metrics):
        """Update metrics display."""
        # Update status labels
        pipeline_metrics = metrics.get('pipeline', {})
        
        self.metric_labels['Packets'].config(
            text=f"Packets: {pipeline_metrics.get('packets_processed', 0)}")
        self.metric_labels['Anomalies'].config(
            text=f"Anomalies: {pipeline_metrics.get('anomalies_detected', 0)}")
        self.metric_labels['Channel Hops'].config(
            text=f"Channel Hops: {pipeline_metrics.get('channel_hops', 0)}")
        self.metric_labels['Processing (ms)'].config(
            text=f"Processing (ms): {pipeline_metrics.get('processing_time_ms', 0):.1f}")
        
        # Update statistics text
        self._update_stats_display(metrics)
        
        # Update channel status
        self._update_channel_display(metrics.get('channel_scanner', {}))
    
    def _update_plot(self):
        """Update real-time plot."""
        if len(self.anomaly_scores) > 0:
            # Get window size
            window = int(self.window_var.get())
            
            # Slice data to window
            x_data = list(self.timestamps)[-window:]
            y_scores = list(self.anomaly_scores)[-window:]
            y_threshold = list(self.threshold_values)[-window:]
            
            # Update plot data
            self.anomaly_line.set_data(x_data, y_scores)
            self.threshold_line.set_data(x_data, y_threshold)
            
            # Adjust limits
            if x_data:
                self.ax.set_xlim(x_data[0], x_data[-1])
                y_min = min(min(y_scores), min(y_threshold)) * 0.9
                y_max = max(max(y_scores), max(y_threshold)) * 1.1
                self.ax.set_ylim(y_min, y_max)
            
            # Redraw
            self.canvas.draw_idle()
    
    def _update_stats_display(self, metrics):
        """Update statistics display."""
        stats_text = "=== Anomaly Detection ===\n"
        ad_metrics = metrics.get('anomaly_detector', {})
        threshold_stats = ad_metrics.get('threshold_stats', {})
        
        stats_text += f"Total Samples: {ad_metrics.get('total_samples', 0)}\n"
        stats_text += f"Anomaly Rate: {ad_metrics.get('anomaly_rate', 0):.2%}\n"
        stats_text += f"Threshold: {threshold_stats.get('threshold', 0):.3f}\n"
        stats_text += f"Mean Score: {threshold_stats.get('mean', 0):.3f}\n"
        stats_text += f"Std Dev: {threshold_stats.get('std', 0):.3f}\n\n"
        
        stats_text += "=== Classification ===\n"
        class_metrics = metrics.get('classifier', {})
        type_counts = class_metrics.get('type_counts', {})
        
        for jtype, count in type_counts.items():
            stats_text += f"{jtype}: {count}\n"
        
        stats_text += f"\nAvg Confidence: {class_metrics.get('average_confidence', 0):.2f}\n"
        
        # Update text widget
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def _update_channel_display(self, channel_data):
        """Update channel status display."""
        # Clear existing
        for item in self.channel_tree.get_children():
            self.channel_tree.delete(item)
        
        # Add channel data
        channels = channel_data.get('channels', {})
        for freq, status in channels.items():
            freq_mhz = f"{freq/1e6:.1f} MHz"
            energy = f"{status['energy_dbm']:.1f}"
            clean = "Clean" if status['is_clean'] else "Occupied"
            last_scan = datetime.fromtimestamp(status['last_scan']).strftime('%H:%M:%S') if status['last_scan'] > 0 else "Never"
            jammed = "Yes" if status['jamming_detected'] else "No"
            
            self.channel_tree.insert('', tk.END, values=(freq_mhz, energy, clean, last_scan, jammed))
    
    def _process_signal(self, signal_data):
        """Process signal data for visualization."""
        try:
            # Use INFO level so we can see what's happening
            logger.info(f"Processing signal data: type={type(signal_data)}")
            if isinstance(signal_data, dict):
                logger.info(f"Dict keys: {list(signal_data.keys())}")
            elif isinstance(signal_data, np.ndarray):
                logger.info(f"Array shape: {signal_data.shape}, dtype: {signal_data.dtype}, complex: {np.iscomplexobj(signal_data)}")
            
            # Extract I/Q data
            if isinstance(signal_data, dict) and 'iq_data' in signal_data:
                iq_samples = signal_data['iq_data']
                logger.info(f"Processing {len(iq_samples)} I/Q samples from dict")
                if len(iq_samples) > 0:
                    # Add to constellation plot data
                    i_data = np.real(iq_samples)
                    q_data = np.imag(iq_samples)
                    
                    # Add to deques (automatically handles maxlen)
                    self.constellation_points['i'].extend(i_data[-100:])  # Keep last 100 points
                    self.constellation_points['q'].extend(q_data[-100:])
                    logger.info(f"Added {len(i_data)} samples to constellation data")
            
            # Extract FFT data for spectrogram
            if isinstance(signal_data, dict) and 'fft_data' in signal_data:
                fft_magnitude = np.abs(signal_data['fft_data'])
                logger.info(f"Processing FFT data with {len(fft_magnitude)} bins")
                # Normalize
                if np.max(fft_magnitude) > 0:
                    fft_magnitude = fft_magnitude / np.max(fft_magnitude)
                
                # Add to spectrogram data
                self.spectrogram_data.append(fft_magnitude)
                logger.info(f"Added FFT data to spectrogram, total frames: {len(self.spectrogram_data)}")
            
            # If signal_data is just raw samples (numpy array), treat as I/Q data
            elif isinstance(signal_data, np.ndarray):
                logger.info(f"Processing raw numpy array with {len(signal_data)} samples")
                if np.iscomplexobj(signal_data):
                    # Complex I/Q data
                    i_data = np.real(signal_data)
                    q_data = np.imag(signal_data)
                    
                    self.constellation_points['i'].extend(i_data[-100:])
                    self.constellation_points['q'].extend(q_data[-100:])
                    
                    # Compute FFT for spectrogram
                    fft_data = np.fft.fft(signal_data)
                    fft_magnitude = np.abs(fft_data)
                    if np.max(fft_magnitude) > 0:
                        fft_magnitude = fft_magnitude / np.max(fft_magnitude)
                    self.spectrogram_data.append(fft_magnitude)
                    logger.info(f"Generated FFT from complex samples")
                else:
                    # Real data - treat as I channel, Q=0
                    i_data = signal_data
                    q_data = np.zeros_like(signal_data)
                    
                    self.constellation_points['i'].extend(i_data[-100:])
                    self.constellation_points['q'].extend(q_data[-100:])
                    
                    # Compute FFT for spectrogram
                    fft_data = np.fft.fft(signal_data)
                    fft_magnitude = np.abs(fft_data)
                    if np.max(fft_magnitude) > 0:
                        fft_magnitude = fft_magnitude / np.max(fft_magnitude)
                    self.spectrogram_data.append(fft_magnitude)
                    logger.info(f"Generated FFT from real samples")
            
            else:
                logger.warning(f"Unknown signal data format: {type(signal_data)}")
        
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_signal_plots(self):
        """Update signal visualization plots."""
        try:
            # Debug logging
            const_i_len = len(self.constellation_points['i'])
            const_q_len = len(self.constellation_points['q'])
            spec_len = len(self.spectrogram_data)
            logger.debug(f"Updating plots: constellation I={const_i_len}, Q={const_q_len}, spectrogram={spec_len}")
            
            # Update constellation plot
            if const_i_len > 0 and const_q_len > 0:
                i_data = list(self.constellation_points['i'])
                q_data = list(self.constellation_points['q'])
                
                # Update scatter plot
                self.const_ax.clear()
                self.const_ax.set_title('I/Q Constellation')
                self.const_ax.set_xlabel('In-phase (I)')
                self.const_ax.set_ylabel('Quadrature (Q)')
                self.const_ax.grid(True, alpha=0.3)
                self.const_ax.set_xlim(-3, 3)
                self.const_ax.set_ylim(-3, 3)
                
                if len(i_data) > 0:
                    self.const_ax.scatter(i_data, q_data, c='blue', alpha=0.5, s=1)
                    logger.debug(f"Updated constellation with {len(i_data)} points")
            
            # Update spectrogram
            if spec_len > 0:
                # Convert to 2D array
                spec_array = np.array(list(self.spectrogram_data))
                logger.debug(f"Spectrogram array shape: {spec_array.shape}")
                
                # Update spectrogram image
                self.spec_ax.clear()
                self.spec_ax.set_title('Spectrogram (FFT Magnitude)')
                self.spec_ax.set_xlabel('Frequency Bin')
                self.spec_ax.set_ylabel('Time')
                
                im = self.spec_ax.imshow(
                    spec_array,
                    aspect='auto',
                    cmap='viridis',
                    interpolation='nearest',
                    origin='lower'
                )
                logger.debug("Updated spectrogram image")
            
            # Update time domain plot
            if const_i_len > 0:
                i_data = list(self.constellation_points['i'])[-500:]  # Last 500 samples
                q_data = list(self.constellation_points['q'])[-500:]
                
                self.time_ax.clear()
                self.time_ax.set_title('Time Domain Signal')
                self.time_ax.set_xlabel('Sample')
                self.time_ax.set_ylabel('Amplitude')
                self.time_ax.grid(True, alpha=0.3)
                
                if len(i_data) > 0:
                    sample_indices = range(len(i_data))
                    self.time_ax.plot(sample_indices, i_data, 'b-', label='I', alpha=0.7)
                    self.time_ax.plot(sample_indices, q_data, 'r-', label='Q', alpha=0.7)
                    self.time_ax.legend()
                    logger.debug(f"Updated time domain with {len(i_data)} samples")
            
            # Update power spectrum
            if spec_len > 0:
                # Use latest FFT data
                latest_fft = self.spectrogram_data[-1]
                psd = 20 * np.log10(latest_fft + 1e-10)  # Convert to dB
                
                self.psd_ax.clear()
                self.psd_ax.set_title('Power Spectral Density')
                self.psd_ax.set_xlabel('Frequency Bin')
                self.psd_ax.set_ylabel('Power (dB)')
                self.psd_ax.grid(True, alpha=0.3)
                
                freq_bins = range(len(psd))
                self.psd_ax.plot(freq_bins, psd, 'g-')
                logger.debug(f"Updated PSD with {len(psd)} bins")
            
            # Redraw canvas
            self.signal_canvas.draw_idle()
            logger.debug("Signal canvas redrawn")
            
        except Exception as e:
            logger.error(f"Signal plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def log_message(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def clear_plot(self):
        """Clear the real-time plot."""
        self.anomaly_scores.clear()
        self.timestamps.clear()
        self.threshold_values.clear()
        self.log_message("Plot cleared")
    
    def clear_log(self):
        """Clear the log."""
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        """Save log to file."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                               filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"Log saved to {filename}")
    
    def start_pipeline(self):
        """Start the processing pipeline."""
        if self.pipeline:
            self.pipeline.start()
            self.status_label.config(text="Status: Running")
            self.log_message("Pipeline started")
    
    def stop_pipeline(self):
        """Stop the processing pipeline."""
        if self.pipeline:
            self.pipeline.stop()
            self.status_label.config(text="Status: Stopped")
            self.log_message("Pipeline stopped")
    
    def reset_threshold(self):
        """Reset adaptive threshold."""
        if self.pipeline:
            self.pipeline.anomaly_detector.reset_threshold()
            self.log_message("Threshold reset")
    
    def channel_sweep(self):
        """Perform channel sweep."""
        if self.pipeline:
            threading.Thread(target=self.pipeline.channel_scanner.perform_channel_sweep,
                           daemon=True).start()
            self.log_message("Channel sweep started")
    
    def save_models(self):
        """Save trained models."""
        from tkinter import filedialog
        directory = filedialog.askdirectory()
        if directory and self.pipeline:
            self.pipeline.save_models(directory)
            self.log_message(f"Models saved to {directory}")
    
    def load_models(self):
        """Load trained models."""
        # Implement model loading dialog
        self.log_message("Model loading not implemented yet")
    
    def on_closing(self):
        """Handle window closing."""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
        self.root.quit()
    
    def generate_test_signal(self):
        """Generate test signal for visualization debugging."""
        logger.info("Generating test signal for visualization")
        
        # Generate test I/Q data
        samples = 1024
        t = np.linspace(0, 1, samples)
        
        # Create a complex signal with multiple frequency components
        freq1, freq2 = 50, 120
        noise_level = 0.1
        
        signal = (np.exp(1j * 2 * np.pi * freq1 * t) + 
                 0.5 * np.exp(1j * 2 * np.pi * freq2 * t) + 
                 noise_level * (np.random.randn(samples) + 1j * np.random.randn(samples)))
        
        # Create test signal data
        test_data = {
            'iq_data': signal,
            'fft_data': np.fft.fft(signal)
        }
        
        # Process through the signal visualization pipeline
        self._process_signal(test_data)
        self._update_signal_plots()
        
        self.log_message("Test signal generated and processed")
    
    def run(self):
        """Start the GUI main loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()